from typing import Any, Dict

from sqlalchemy import bindparam, or_, select
from sqlalchemy.sql import Select

from execution_engine.constants import CohortCategory, OMOPConcepts
from execution_engine.omop.criterion.concept import ConceptCriterion

__all__ = ["VisitOccurrence", "ActivePatients", "PatientsActiveDuringPeriod"]


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

    def _sql_header(
        self, distinct_person: bool = True, person_id: int | None = None
    ) -> Select:
        """
        Get the SQL header for the criterion.
        """

        c = self._table.c.person_id.label("person_id")

        # if distinct_person:
        #    c = distinct(c)

        query = select(c).select_from(self._table)

        return query

    def _sql_generate(self, query: Select) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        query = query.filter(
            self._table.c.visit_type_concept_id
            == OMOPConcepts.VISIT_TYPE_STILL_PATIENT.value
        )

        return query

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


class PatientsActiveDuringPeriod(ActivePatients):
    """
    Select Patients who were hospitalized during a given period
    """

    def _sql_generate(self, query: Select) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        query = query.filter(
            or_(
                self._table.c.visit_start_datetime.between(
                    bindparam("observation_start_datetime"),
                    bindparam("observation_end_datetime"),
                ),
                self._table.c.visit_end_datetime.between(
                    bindparam("observation_start_datetime"),
                    bindparam("observation_end_datetime"),
                ),
            )
        )

        return query
