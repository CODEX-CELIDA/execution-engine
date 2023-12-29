from typing import Any, Dict

from sqlalchemy.sql import Select, extract

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import (
    column_interval_type,
    create_conditional_interval_column,
)
from execution_engine.omop.criterion.continuous import ContinuousCriterion
from execution_engine.util import Interval, ValueNumber, value_factory
from execution_engine.util.interval import IntervalType

__all__ = ["ProcedureOccurrence"]


class ProcedureOccurrence(ContinuousCriterion):
    """A procedure occurrence criterion in a recommendation."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        concept: Concept,
        value: ValueNumber | None = None,
        timing: ValueNumber | None = None,
        static: bool | None = None,
    ) -> None:
        super().__init__(
            name=name,
            exclude=exclude,
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

        start_datetime = self._table.c["procedure_datetime"]
        end_datetime = self._table.c["procedure_end_datetime"]

        query = self._sql_header()
        query = self._sql_filter_concept(query)

        # todo: is this even required in procedure?
        if self._value is not None:
            conditional_column = create_conditional_interval_column(
                self._value.to_sql(self._table)
            )
        else:
            conditional_column = column_interval_type(IntervalType.POSITIVE)

        query = query.add_columns(conditional_column)

        # todo: this should not filter but also set the interval_type
        if self._timing is not None:
            interval = Interval(self._timing.unit.concept_code)
            column = extract(interval.name, end_datetime - start_datetime).label(
                "duration"
            )
            query = query.filter(
                self._timing.to_sql(table=None, column_name=column, with_unit=False)
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
        return f"{self.__class__.__name__}['{self._name}'](concept={self._concept.concept_name}, value={str(self._value)}, timing={str(self._timing)})"

    def dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the criterion.
        """
        return {
            "name": self._name,
            "exclude": self._exclude,
            "category": self._category.value,
            "concept": self._concept.dict(),
            "value": self._value.dict() if self._value is not None else None,
            "timing": self._timing.dict() if self._timing is not None else None,
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
            isinstance(timing, ValueNumber) or timing is None
        ), "timing must be a ValueNumber"

        return cls(
            name=data["name"],
            exclude=data["exclude"],
            category=CohortCategory(data["category"]),
            concept=Concept(**data["concept"]),
            value=value,
            timing=timing,
        )
