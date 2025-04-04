from abc import ABC

from sqlalchemy.sql import Select

from execution_engine.constants import OMOPConcepts
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.util.types import Timing
from execution_engine.util.value import Value

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


class ConceptCriterion(Criterion, ABC):
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
        concept: Concept,
        value: Value | None = None,
        static: bool | None = None,
        timing: Timing | None = None,
        value_required: bool | None = None,
    ):
        super().__init__()

        assert (
            isinstance(value, Value) or value is None
        ), f"Value must be None or an instance of Value, got {type(value)} instead."

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

        if value_required is not None and isinstance(value_required, bool):
            self._value_required = value_required

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
