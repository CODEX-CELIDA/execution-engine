from typing import Any, Dict, cast

from sqlalchemy import case, func, select
from sqlalchemy.sql import Select

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import (
    SQL_ONE_SECOND,
    column_interval_type,
    create_conditional_interval_column,
)
from execution_engine.omop.criterion.continuous import ContinuousCriterion
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import Timing
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.factory import value_factory

__all__ = ["ProcedureOccurrence"]


class ProcedureOccurrence(ContinuousCriterion):
    """A procedure occurrence criterion in a recommendation."""

    def __init__(
        self,
        category: CohortCategory,
        concept: Concept,
        value: ValueNumber | None = None,
        timing: Timing | None = None,
        static: bool | None = None,
    ) -> None:
        super().__init__(
            category=category,
            concept=concept,
            value=value,
            static=static,
        )

        self._set_omop_variables_from_domain("procedure")
        self._timing = timing

    def _create_query(
        self,
    ) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        query = select(
            self._table.c.person_id,
        ).select_from(self._table)
        query = self._sql_filter_concept(query)
        query = self._filter_datetime(query)
        query = self._filter_base_persons(query)

        c_start_datetime = self._table.c["procedure_datetime"]
        c_end_datetime = self._table.c["procedure_end_datetime"]

        if self._timing is not None:
            frequency = self._timing.frequency
            duration = self._timing.duration

            if self._timing.interval is not None:
                assert (
                    frequency is not None
                ), "Frequency must be specified if interval is specified"
                # an interval and a frequency are specified : the query is constructed such that
                # it yields intervals of the specific length (in self._timing.interval) and checks that within each interval
                # the frequency is satisfied

                interval = self._timing.interval.to_sql_interval()
                interval_starts = self.cte_interval_starts(query, interval)
                date_ranges = self.cte_date_ranges(interval_starts, interval)

                # if a duration is specified, we need to count the number of occurrences within each interval
                if duration is not None:
                    # make sure that the duration from the query is in the same time unit (e.g. h) as desired duration
                    c_interval_count = func.sum(
                        case(
                            (duration.to_sql(column_name=date_ranges.c.duration), 1),
                            else_=0,
                        )
                    ).label("interval_count")
                else:
                    c_interval_count = func.count().label("interval_count")

                # interval_type is determined by the number of occurrences within each interval (w.r.t. the desired
                # frequency)
                conditional_interval_column = create_conditional_interval_column(
                    condition=frequency.to_sql(column_name=c_interval_count)
                )

                query = (
                    select(
                        date_ranges.c.person_id,
                        date_ranges.c.interval_start.label("interval_start"),
                        (
                            date_ranges.c.interval_start + interval - SQL_ONE_SECOND
                        ).label("interval_end"),
                        conditional_interval_column.label("interval_type"),
                    )
                    .select_from(date_ranges)
                    .group_by(date_ranges.c.person_id, date_ranges.c.interval_start)
                )
            elif duration is not None:
                c_duration = (c_end_datetime - c_start_datetime).label("duration")
                conditional_column = create_conditional_interval_column(
                    duration.to_sql(table=None, column_name=c_duration)
                )

                query = query.add_columns(
                    conditional_column,
                    c_start_datetime.label("interval_start"),
                    c_end_datetime.label("interval_end"),
                )
            elif self._timing.count is not None:
                raise NotImplementedError("Count timing not implemented yet")
            else:
                raise ValueError("Timing must have an interval, duration or count")
        else:
            # no timing is given, so we just need to check that the procedure is performed
            conditional_column = column_interval_type(IntervalType.POSITIVE)

            query = query.add_columns(
                conditional_column,
                c_start_datetime.label("interval_start"),
                c_end_datetime.label("interval_end"),
            )

        return query

    def _sql_select_data(self, query: Select) -> Select:
        query = query.add_columns(
            self._table.c["procedure_concept_id"].label("parameter_concept_id"),
            self._table.c["procedure_datetime"].label("start_datetime"),
            self._table.c["procedure_end_datetime"].label("end_datetime"),
        )

        return query

    def description(self) -> str:
        """
        Get a human-readable description of the criterion.
        """

        assert self._concept is not None, "Concept must be set"

        parts = [f"concept={self._concept.concept_name}"]
        if self._timing is not None:
            parts.append(f"dose={str(self._timing)}")

        return f"{self.__class__.__name__}[" + ", ".join(parts) + "]"

    def dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the criterion.
        """
        assert self._concept is not None, "Concept must be set"

        return {
            "category": self._category.value,
            "concept": self._concept.model_dump(),
            "value": (
                self._value.model_dump(include_meta=True)
                if self._value is not None
                else None
            ),
            "timing": (
                self._timing.model_dump(include_meta=True)
                if self._timing is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcedureOccurrence":
        """
        Create a procedure occurrence criterion from a dictionary representation.
        """

        value = value_factory(**data["value"]) if data["value"] is not None else None
        timing = value_factory(**data["timing"]) if data["timing"] is not None else None

        assert (
            isinstance(value, ValueNumber) or value is None
        ), "value must be a ValueNumber"
        assert (
            isinstance(timing, ValueNumber | Timing) or timing is None
        ), "timing must be a ValueNumber"

        return cls(
            category=CohortCategory(data["category"]),
            concept=Concept(**data["concept"]),
            value=value,
            timing=cast(Timing, timing),
        )
