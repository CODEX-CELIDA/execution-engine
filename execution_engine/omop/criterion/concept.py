from datetime import datetime

from ...util import Value
from ..concepts import Concept
from .abstract import Criterion


class ConceptCriterion(Criterion):
    """
    Abstract class for a criterion based on an OMOP concept and optional value.

    This class is not meant to be instantiated directly. Instead, use one of the subclasses.
    Subclasses need to set _OMOP_TABLE and _OMOP_COLUMN_PREFIX (e.g. "visit_occurrence" and "visit").
    These identify the base table in OMOP CDM and the prefix of the concept_id column (e.g. "visit_concept_id").

    """

    _OMOP_TABLE: str
    _OMOP_COLUMN_PREFIX: str

    def __init__(
        self,
        name: str,
        concept: Concept,
        exclude: bool = False,
        value: Value | None = None,
    ):
        super().__init__(name, exclude)
        self._concept = concept
        self._value = value
        self._table_in: str | None = None
        self._table_out: str | None = None
        self._start_datetime: datetime | None = None
        self._end_datetime: datetime | None = None

    def _sql_generate(self, sql_select: str) -> str:
        """
        Get the SQL representation of the criterion.
        """
        sql = sql_select
        sql += (
            f"INNER JOIN {self._OMOP_TABLE} co ON (co.person_id = table_in.person_id)\n"
            f"WHERE {self._OMOP_COLUMN_PREFIX}_concept_id = {self._concept.id}\n"
        )

        return sql
