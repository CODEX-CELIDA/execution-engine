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
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def _sql_interval_type_column(self, query: Select | CTE) -> ColumnElement:
        """
        Add the value to the SQL query.

        :param query: The SQL query.
        """
        if self._value is not None:
            conditional_column = create_conditional_interval_column(
                self._value.to_sql(table=query, with_unit=self._value.supports_units())
            )
        else:
            conditional_column = column_interval_type(IntervalType.POSITIVE)

        return conditional_column.label("interval_type")

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        # todo: handle self._static variables - these should be backfilled within the observation range ?

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

        Forward fill all intervals and in insert NO_DATA intervals for missing time in observation_window.

        :param df: The result of the SQL query.
        :param base_data: The base data.
        :param observation_window: The observation window.
        :return: A processed DataFrame.
        """
        df = process.forward_fill(df)

        no_data_intervals = process.complementary_intervals(
            df,
            reference_df=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NO_DATA,
        )
        df = pd.concat([df, no_data_intervals])
        df.sort_values(by=["person_id", "interval_start"], inplace=True)

        return df
