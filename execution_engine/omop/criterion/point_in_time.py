from sqlalchemy import Interval, Select, bindparam, case, func, select

from execution_engine.constants import IntervalType
from execution_engine.omop.criterion.abstract import column_interval_type
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.db.celida.tables import IntervalTypeEnum


class PointInTimeCriterion(ConceptCriterion):
    """A point-in-time criterion in a cohort definition."""

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        if self._OMOP_VALUE_REQUIRED:
            assert self._value is not None, "Value is required for this criterion"

        interval_hours_param = bindparam("validity_threshold_hours", value=12)
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
            conditional_column = case(
                (
                    self._value.to_sql(cte),
                    IntervalType.POSITIVE,
                ),
                else_=IntervalType.NEGATIVE,
            ).cast(IntervalTypeEnum)
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
