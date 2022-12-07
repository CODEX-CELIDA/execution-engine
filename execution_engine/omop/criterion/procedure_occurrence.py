from sqlalchemy import literal_column
from sqlalchemy.sql import extract

from ...util import ValueNumber, ucum_to_postgres
from ...util.sql import SelectInto
from ..concepts import Concept
from .concept import ConceptCriterion


class ProcedureOccurrence(ConceptCriterion):
    """A procedure occurrence criterion in a cohort definition."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        concept: Concept,
        value: ValueNumber | None = None,
        timing: ValueNumber | None = None,
    ) -> None:
        super().__init__(name=name, exclude=exclude, concept=concept, value=value)

        self._set_omop_variables_from_domain("procedure")
        self._timing = timing

    def _sql_generate(self, base_sql: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        sql = base_sql

        concept_id = literal_column(f"{self._OMOP_COLUMN_PREFIX}_concept_id")
        start_datetime = literal_column(f"{self._OMOP_COLUMN_PREFIX}_datetime")
        end_datetime = literal_column(f"{self._OMOP_COLUMN_PREFIX}_end_datetime")

        tbl_join = self._OMOP_TABLE.alias(self.table_alias)

        sql = sql.join(
            tbl_join,
            tbl_join.c.person_id == self._table_in.c.person_id,
        ).filter(tbl_join.columns.corresponding_column(concept_id) == self._concept.id)

        if self._timing is not None:
            interval = ucum_to_postgres[self._timing.unit.concept_code]
            sql = sql.add_columns(
                extract(interval, start_datetime - end_datetime).label("duration")
            )
            sql = sql.filter(
                self._timing.to_sql(
                    table_name=self.table_alias, column_name="duration", with_unit=False
                )
            )

        return base_sql
