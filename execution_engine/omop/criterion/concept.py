from datetime import datetime

from sqlalchemy.sql import TableClause

from ...util import Value
from ...util.sql import SelectInto
from ..concepts import Concept
from .abstract import Criterion


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
        concept: Concept,
        value: Value | None = None,
    ):
        super().__init__(name, exclude)

        self._set_omop_variables_from_domain(concept.domain_id)
        self._concept = concept
        self._value = value
        self._table_in: TableClause
        self._table_out: TableClause
        self._start_datetime: datetime | None = None
        self._end_datetime: datetime | None = None

    def _sql_generate(self, base_sql: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        if self._OMOP_VALUE_REQUIRED and self._value is None:
            raise ValueError(
                f'Value must be set for "{self._OMOP_TABLE.__tablename__}" criteria'
            )

        sql = base_sql.select

        concept_column_name = f"{self._OMOP_COLUMN_PREFIX}_concept_id"

        sql = sql.join(
            self._table_join,
            self._table_join.c.person_id == self._table_in.c.person_id,
        ).filter(self._table_join.c[concept_column_name] == self._concept.id)

        if self._value is not None:
            sql = sql.filter(self._value.to_sql(self.table_alias))

        base_sql.select = sql

        return base_sql
