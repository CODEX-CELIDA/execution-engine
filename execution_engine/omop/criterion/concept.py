from typing import Any, Dict

from sqlalchemy.sql import Select

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.util import Value, value_factory

__all__ = ["ConceptCriterion"]


class ConceptCriterion(Criterion):
    """
    Abstract class for a criterion based on an OMOP concept and optional value.

    This class is not meant to be instantiated directly. Instead, use one of the subclasses.
    Subclasses need to set _OMOP_TABLE and _OMOP_COLUMN_PREFIX (e.g. "visit_occurrence" and "visit").
    These identify the base table in OMOP CDM and the prefix of the concept_id column (e.g. "visit_concept_id").

    """

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        concept: Concept,
        value: Value | None = None,
    ):
        super().__init__(name=name, exclude=exclude, category=category)

        self._set_omop_variables_from_domain(concept.domain_id)
        self._concept = concept
        self._value = value

    def _sql_generate(self, sql: Select) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        if self._OMOP_VALUE_REQUIRED and self._value is None:
            raise ValueError(
                f'Value must be set for "{self._OMOP_TABLE.__tablename__}" criteria'
            )

        concept_column_name = f"{self._OMOP_COLUMN_PREFIX}_concept_id"

        sql = sql.filter(self._table.c[concept_column_name] == self._concept.concept_id)

        if self._value is not None:
            sql = sql.filter(self._value.to_sql(self.table_alias))

        return sql

    def _sql_select_data(self, query: Select) -> Select:
        c_start = self._get_datetime_column(self._table)
        concept_column_name = f"{self._OMOP_COLUMN_PREFIX}_concept_id"

        query = query.add_columns(
            self._table.c[concept_column_name].label("parameter_concept_id")
        )
        query = query.add_columns(c_start.label("start_datetime"))

        if self._value is not None:
            # need to add value column to select (and unit !)
            import warnings

            warnings.warn("# need to add value column to select (and unit !)")

        return query

    def dict(self) -> dict[str, Any]:
        """
        Get a JSON representation of the criterion.
        """
        return {
            "name": self._name,
            "exclude": self._exclude,
            "category": self._category.value,
            "concept": self._concept.dict(),
            "value": self._value.dict() if self._value is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptCriterion":
        """
        Create a criterion from a JSON representation.
        """

        return cls(
            name=data["name"],
            exclude=data["exclude"],
            category=CohortCategory(data["category"]),
            concept=Concept(**data["concept"]),
            value=value_factory(**data["value"]) if data["value"] is not None else None,
        )
