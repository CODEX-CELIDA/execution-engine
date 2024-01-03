from typing import Any

import pandas as pd
from sqlalchemy import CTE, ColumnElement, Select, select

from execution_engine.omop.criterion.abstract import (
    column_interval_type,
    create_conditional_interval_column,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.task import process
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import TimeRange


class PointInTimeCriterion(ConceptCriterion):
    """A point-in-time criterion in a recommendation."""

    def __init__(
        self,
        *args: Any,
        validity_duration_hours: float | None = 12,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._validity_duration_hours: float | None = None

        if self._static:
            self._validity_duration_hours = float("inf")
        else:
            self._validity_duration_hours = validity_duration_hours

    def _sql_interval_type_column(self, query: Select | CTE) -> ColumnElement:
        """
        Add the value to the SQL query.

        :param query: The SQL query.
        """
        if self._value is not None:
            conditional_column = create_conditional_interval_column(
                self._value.to_sql(table=query)
            )
        else:
            conditional_column = column_interval_type(IntervalType.POSITIVE)

        return conditional_column.label("interval_type")

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        if self._OMOP_VALUE_REQUIRED:
            assert self._value is not None, "Value is required for this criterion"

        datetime_col = self._get_datetime_column(self._table, "start")

        cte = select(
            self._table.c.person_id,
            datetime_col.label("interval_start"),
            datetime_col.label("interval_end"),
            self._table.c.value_as_concept_id,
            self._table.c.value_as_number,
            self._table.c.unit_concept_id,
        )
        cte = self._sql_filter_concept(cte).cte("measurements")

        conditional_column = self._sql_interval_type_column(cte)

        query = select(
            cte.c.person_id,
            conditional_column,
            cte.c.interval_start,
            cte.c.interval_end,
        )

        return query

    def process_result(
        self, df: pd.DataFrame, base_data: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Process the result of the SQL query.

        Inserts NO_DATA intervals for all intervals that are not POSITIVE or NEGATIVE.

        :param df: The result of the SQL query.
        :param base_data: The base data.
        :param observation_window: The observation window.
        :return: A processed DataFrame.
        """
        no_data_intervals = process.complementary_intervals(
            df,
            reference_df=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NO_DATA,
        )
        df = pd.concat([df, no_data_intervals])
        df.sort_values(by=["person_id", "interval_start"], inplace=True)

        return df
