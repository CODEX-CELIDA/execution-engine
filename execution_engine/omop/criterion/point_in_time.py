import pandas as pd
from sqlalchemy import Interval, Select, bindparam, func, select

from execution_engine.constants import IntervalType
from execution_engine.omop.criterion.abstract import (
    column_interval_type,
    create_conditional_interval_column,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.task import process
from execution_engine.util import TimeRange


class PointInTimeCriterion(ConceptCriterion):
    """A point-in-time criterion in a cohort definition."""

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        if self._OMOP_VALUE_REQUIRED:
            assert self._value is not None, "Value is required for this criterion"

        interval_hours_param = bindparam(
            "validity_threshold_hours", value=12
        )  # todo make dynamic
        datetime_col = self._get_datetime_column(self._table, "start")
        time_threshold_param = func.cast(
            func.concat(interval_hours_param, "hours"), Interval
        )

        cte = select(
            self._table.c.person_id,
            datetime_col.label("datetime"),
            self._table.c.value_as_concept_id,
            self._table.c.value_as_number,
            self._table.c.unit_concept_id,
            func.lead(datetime_col)
            .over(partition_by=self._table.c.person_id, order_by=datetime_col)
            .label("next_datetime"),
        )
        cte = self._sql_filter_concept(cte).cte("RankedMeasurements")

        if self._value is not None:
            conditional_column = create_conditional_interval_column(
                self._value.to_sql(table=cte)
            )
        else:
            conditional_column = column_interval_type(IntervalType.POSITIVE)

        query = select(
            cte.c.person_id,
            conditional_column.label("interval_type"),
            cte.c.datetime.label("interval_start"),
            func.least(
                cte.c.datetime + time_threshold_param,
                func.coalesce(
                    cte.c.next_datetime, cte.c.datetime + time_threshold_param
                ),
            ).label("interval_end"),
        )

        return query

    def process_result(
        self, df: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Process the result of the SQL query.

        Inserts NO_DATA intervals for all intervals that are not POSITIVE or NEGATIVE.

        :param df: The result of the SQL query.
        :return: A processed DataFrame.
        """
        no_data_intervals = process.invert_intervals(
            df, ["person_id"], observation_window
        )
        no_data_intervals["interval_type"] = IntervalType.NO_DATA
        df = pd.concat([df, no_data_intervals])
        df.sort_values(by=["person_id", "interval_start"], inplace=True)

        return df
