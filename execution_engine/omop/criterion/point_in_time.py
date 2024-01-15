from typing import Any

import pandas as pd
from sqlalchemy import CTE, ColumnElement, Select, select

from execution_engine.omop.criterion.abstract import (
    column_interval_type,
    create_conditional_interval_column,
    observation_end_datetime,
    observation_start_datetime,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.task import process_rect as process
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import PersonIntervals, TimeRange


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

        if self._OMOP_VALUE_REQUIRED:
            assert self._value is not None, "Value is required for this criterion"

        if self._static:
            # If this criterion is a static criterion, i.e. one whose value does not change over time,
            # then we add the observation range as the interval range
            c_start = observation_start_datetime
            c_end = observation_end_datetime
        else:
            datetime_col = self._get_datetime_column(self._table, "start")
            c_start, c_end = datetime_col, datetime_col

        cte = select(
            self._table.c.person_id,
            c_start.label("interval_start"),
            c_end.label("interval_end"),
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

    def process_data(
        self,
        data: PersonIntervals,
        base_data: pd.DataFrame,
        observation_window: TimeRange,
    ) -> PersonIntervals:
        """
        Process the result of the SQL query.

        Forward fill all intervals and in insert NO_DATA intervals for missing time in observation_window.

        :param data: The result of the SQL query.
        :param base_data: The base data.
        :param observation_window: The observation window.
        :return: A processed DataFrame.
        """
        data = process.forward_fill(data)

        no_data_intervals = process.complementary_intervals(
            data,
            reference=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NO_DATA,
        )
        data = process.concat_intervals([data, no_data_intervals])

        return data
