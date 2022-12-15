from typing import Any, Dict

from sqlalchemy import literal_column
from sqlalchemy.sql import Insert

from ...constants import CohortCategory
from ...util.sql import SelectInto
from .. import StandardConcepts
from .concept import ConceptCriterion

__all__ = ["VisitOccurrence", "ActivePatients"]


class VisitOccurrence(ConceptCriterion):
    """A visit criterion in a cohort definition."""


class ActivePatients(VisitOccurrence):
    """
    Select only patients who are still hospitalized.
    """

    def __init__(self, name: str):
        self._name = name
        self._exclude = False
        self._category = CohortCategory.BASE
        self._set_omop_variables_from_domain("visit")

    def _sql_header(self, table_in: str | None, table_out: str) -> Insert:
        """
        Get the SQL header for the criterion.
        """
        if table_in is not None:
            raise ValueError("ActivePatients must be the first criterion")

        return super()._sql_header(self._OMOP_TABLE, table_out)

    def _sql_generate(self, base_sql: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        sql = base_sql.select
        sql = sql.filter(
            literal_column("visit_type_concept_id")
            == StandardConcepts.VISIT_TYPE_STILL_PATIENT.value
        )

        base_sql.select = sql

        return base_sql

    def dict(self) -> dict[str, Any]:
        """
        Get a JSON representation of the criterion.
        """
        return {"class_name": self.__class__.__name__, "data": {"name": self._name}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivePatients":
        """
        Create a criterion from a JSON representation.
        """
        return cls(data["name"])
