from sqlalchemy import literal_column, table
from sqlalchemy.sql import Insert, Select

from ...util.sql import SelectInto
from .. import StandardConcepts
from .concept import ConceptCriterion


class VisitOccurrence(ConceptCriterion):
    """A visit criterion in a cohort definition."""

    _static: bool = True


class ActivePatients(VisitOccurrence):
    """
    Select only patients who are still hospitalized.
    """

    def __init__(self, name: str):
        self._name = name
        self._exclude = False
        self._set_omop_variables_from_domain("visit")

    def _sql_header(self, table_in: str | None, table_out: str) -> Insert:
        """
        Get the SQL header for the criterion.
        """
        if table_in is not None:
            raise ValueError("ActivePatients must be the first criterion")

        return super()._sql_header(self._OMOP_TABLE, table_out)

    def _sql_generate(self, sql_header: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        sql_header = sql_header.filter(
            literal_column("visit_type_concept_id")
            == StandardConcepts.VISIT_TYPE_STILL_PATIENT.value
        )

        return sql_header
