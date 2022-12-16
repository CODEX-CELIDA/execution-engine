from typing import Any, Dict

from sqlalchemy.sql import Select, extract

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.util import ValueNumber, ucum_to_postgres, value_factory

pass

__all__ = ["ProcedureOccurrence"]


class ProcedureOccurrence(ConceptCriterion):
    """A procedure occurrence criterion in a cohort definition."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        concept: Concept,
        value: ValueNumber | None = None,
        timing: ValueNumber | None = None,
    ) -> None:
        super().__init__(
            name=name, exclude=exclude, category=category, concept=concept, value=value
        )

        self._set_omop_variables_from_domain("procedure")
        self._timing = timing

    def _sql_generate(self, query: Select) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        concept_id = self._table.c["procedure_concept_id"]
        start_datetime = self._table.c["procedure_datetime"]
        end_datetime = self._table.c["procedure_end_datetime"]

        query = query.filter(concept_id == self._concept.concept_id)

        if self._value is not None:
            query = query.filter(self._value.to_sql(self.table_alias))

        if self._timing is not None:
            interval = ucum_to_postgres[self._timing.unit.concept_code]
            column = extract(interval, start_datetime - end_datetime).label("duration")
            query = query.add_columns(column)
            query = query.filter(
                self._timing.to_sql(
                    table_name=None, column_name=column, with_unit=False
                )
            )

        return query

    def _sql_select_data(self, query: Select) -> Select:

        import warnings

        concept_id = self._table.c["procedure_concept_id"]
        start_datetime = self._table.c["procedure_datetime"]
        end_datetime = self._table.c["procedure_end_datetime"]

        query = query.add_columns(concept_id.label("parameter_concept_id"))
        query = query.add_columns(start_datetime.label("start_datetetime"))
        query = query.add_columns(end_datetime.label("end_datetetime"))

        if self._value is not None:
            warnings.warn("# need to add value column to select (and unit !)")

        if self._timing is not None:
            # need to add value column to select (and unit !)
            warnings.warn("# need to add value column to select (and unit !)")

        # return query

        raise NotImplementedError()

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
