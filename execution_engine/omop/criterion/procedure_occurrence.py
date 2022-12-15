from typing import Any, Dict

from sqlalchemy.sql import extract

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.util import ValueNumber, ucum_to_postgres, value_factory
from execution_engine.util.sql import SelectInto

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

    def _sql_generate(self, base_sql: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        import warnings

        warnings.warn("Make sure that base table is joined for subselects")

        sql = base_sql.select

        concept_id = self._table_join.c["procedure_concept_id"]
        start_datetime = self._table_join.c["procedure_datetime"]
        end_datetime = self._table_join.c["procedure_end_datetime"]

        sql = sql.join(
            self._table_join,
            self._table_join.c.person_id == self._table_in.c.person_id,
        ).filter(concept_id == self._concept.concept_id)

        if self._timing is not None:
            interval = ucum_to_postgres[self._timing.unit.concept_code]
            column = extract(interval, start_datetime - end_datetime).label("duration")
            sql = sql.add_columns(column)
            sql = sql.filter(
                self._timing.to_sql(
                    table_name=None, column_name=column, with_unit=False
                )
            )

        base_sql.select = sql

        return base_sql

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
