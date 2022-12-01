from .. import StandardConcepts
from .concept import ConceptCriterion


class VisitOccurrence(ConceptCriterion):
    """A visit criterion in a cohort definition."""

    _OMOP_TABLE = "visit_occurrence"
    _OMOP_COLUMN_PREFIX = "visit"


class ActivePatients(VisitOccurrence):
    """
    Select only patients who are still hospitalized.
    """

    def __init__(self, name: str):
        self._name = name
        self._exclude = False

    def _sql_generate(self, sql_header: str) -> str:
        """
        Get the SQL representation of the criterion.
        """
        if self._table_in is not None:
            raise ValueError("ActivePatients must be the first criterion")
        sql = self._sql_header(self._OMOP_TABLE, self.table_out)
        sql += f"""
        WHERE visit_type_concept_id = {StandardConcepts.VISIT_TYPE_STILL_PATIENT.value}
        """
        return sql
