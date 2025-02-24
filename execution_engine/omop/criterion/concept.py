from typing import Any, Dict, cast

from sqlalchemy.sql import Select

from execution_engine.constants import CohortCategory, OMOPConcepts
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.meta import SignatureReprMeta
from execution_engine.util.types import Timing
from execution_engine.util.value import Value
from execution_engine.util.value.factory import value_factory

__all__ = ["ConceptCriterion"]

"""
Collection of static clinical variables.

The values of these variables are considered constant over the observation period.
"""
STATIC_CLINICAL_CONCEPTS = [
    int(OMOPConcepts.BODY_WEIGHT_LOINC.value),
    int(OMOPConcepts.BODY_WEIGHT_SNOMED.value),
    int(OMOPConcepts.BODY_HEIGHT.value),
]  # type: list[int]
# TODO: Only use weight etc from the current encounter/visit!


class ConceptCriterion(Criterion, metaclass=SignatureReprMeta):
    """
    Abstract class for a criterion based on an OMOP concept and optional value.

    This class is not meant to be instantiated directly. Instead, use one of the subclasses.
    Subclasses need to set _OMOP_TABLE and _OMOP_COLUMN_PREFIX (e.g. "visit_occurrence" and "visit").
    These identify the base table in OMOP CDM and the prefix of the concept_id column (e.g. "visit_concept_id").

    """

    _concept: Concept
    _value = None
    _timing = None

    def __init__(
        self,
        category: CohortCategory,
        concept: Concept,
        value: Value | None = None,
        static: bool | None = None,
        timing: Timing | None = None,
        override_value_required: bool | None = None,
    ):
        super().__init__(category=category)

        self._set_omop_variables_from_domain(concept.domain_id)
        self._concept = concept
        self._value = value
        self._timing = timing

        # static is a boolean that indicates whether the criterion is static or not
        # it is initially set by the _set_omop_variables_from_domain() function, but can be overridden
        # by supplying a value to the "static" parameter
        if static is not None:
            self._static = static
        else:
            # if static is None, then the criterion is static if the concept is in the STATIC_CLINICAL_CONCEPTS list
            self._static = concept.concept_id in STATIC_CLINICAL_CONCEPTS

        if override_value_required is not None and isinstance(
            override_value_required, bool
        ):
            self._OMOP_VALUE_REQUIRED = override_value_required

    @property
    def concept(self) -> Concept:
        """Get the concept associated with this Criterion"""
        return self._concept

    @property
    def value(self) -> Value | None:
        """Get the value associated with this Criterion"""
        return self._value

    @property
    def timing(self) -> Timing | None:
        """Get the timing associated with this Criterion"""
        return self._timing

    def _sql_filter_concept(
        self, query: Select, override_concept_id: int | None = None
    ) -> Select:
        concept_column_name = f"{self._OMOP_COLUMN_PREFIX}_concept_id"

        concept_id = (
            override_concept_id
            if override_concept_id is not None
            else self._concept.concept_id
        )

        return query.filter(self._table.c[concept_column_name] == concept_id)

    def _sql_select_data(self, query: Select) -> Select:
        c_start = self._get_datetime_column(self._table, "start")

        concept_column_name = f"{self._OMOP_COLUMN_PREFIX}_concept_id"

        query = query.add_columns(
            self._table.c[concept_column_name].label("parameter_concept_id"),
            c_start.label("start_datetime"),
        )

        try:
            c_end = self._get_datetime_column(self._table, "end")
            query = query.add_columns(c_end.label("end_datetime"))
        except ValueError:
            pass  # no end_datetime column

        if self._value is not None:
            query = query.add_columns(self._table.c["value_as_number"])

        return query

    def description(self) -> str:
        """
        Get a human-readable description of the criterion.
        """
        desc = f"{self.__class__.__name__}[concept={self._concept.concept_name}"

        if self._value is not None:
            desc += f", value={str(self._value)}"
        desc += "]"

        return desc

    def dict(self) -> dict[str, Any]:
        """
        Get a JSON representation of the criterion.
        """
        return {
            "category": self._category.value,
            "concept": self._concept.model_dump(),
            "value": (
                self._value.model_dump(include_meta=True)
                if self._value is not None
                else None
            ),
            "static": self._static,
            "timing": (
                self._timing.model_dump(include_meta=True)
                if self._timing is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptCriterion":
        """
        Create a criterion from a JSON representation.
        """

        return cls(
            category=CohortCategory(data["category"]),
            concept=Concept(**data["concept"]),
            value=(
                cast(Value, value_factory(**data["value"]))
                if data["value"] is not None
                else None
            ),
            static=data["static"],
            timing=Timing(**data["timing"]) if data["timing"] is not None else None,
        )
