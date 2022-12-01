from .concept import ConceptCriterion


class Measurement(ConceptCriterion):
    """A measurement criterion in a cohort definition."""

    _OMOP_TABLE = "measurement"
    _OMOP_COLUMN_PREFIX = "measurement"

    def _sql_generate(self, sql_select: str) -> str:
        """
        Get the SQL representation of the criterion.
        """

        if self._value is None:
            raise ValueError("Value must be set for measurement criteria")

        table_alias = "co"

        sql = sql_select
        sql += (
            f"INNER JOIN {self._OMOP_TABLE} {table_alias} ON (co.person_id = table_in.person_id)\n"
            f"WHERE ({self._OMOP_COLUMN_PREFIX}_concept_id = {self._concept.id})\n"
            f" AND ({self._value.to_sql(table_alias)})\n"
        )

        return sql
