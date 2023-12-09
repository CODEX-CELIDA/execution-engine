from sqlalchemy import Interval, Select, bindparam, case, func, select

from execution_engine.constants import IntervalType
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.db.celida.tables import IntervalTypeEnum


class PointInTimeCriterion(ConceptCriterion):
    """A point-in-time criterion in a cohort definition."""

    def create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        assert self._value is not None, "Value must be set for point-in-time criteria"

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

        query = select(
            cte.c.person_id,
            case(
                (self._value.to_sql(cte), IntervalType.POSITIVE),
                else_=IntervalType.NEGATIVE,
            )
            .cast(IntervalTypeEnum)
            .label("interval_type"),
            cte.c.datetime.label("interval_start"),
            func.least(
                cte.c.datetime + time_threshold_param,
                func.coalesce(
                    cte.c.next_datetime, cte.c.datetime + time_threshold_param
                ),
            ).label("interval_end"),
        )

        return query
