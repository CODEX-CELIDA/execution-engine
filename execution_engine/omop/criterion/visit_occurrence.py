from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.sql import Select

from execution_engine.constants import OMOPConcepts
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import column_interval_type
from execution_engine.settings import get_config
from execution_engine.util.interval import IntervalType

__all__ = ["VisitOccurrence", "ActivePatients", "PatientsActiveDuringPeriod"]

from execution_engine.omop.criterion.continuous import ContinuousCriterion


class VisitOccurrence(ContinuousCriterion):
    """A visit criterion in a Recommendation."""


class ActivePatients(VisitOccurrence):
    """
    Select only patients who are still hospitalized.
    """

    def __init__(self) -> None:

        self.set_base(True)  # this is a base criterion

        if get_config().episode_of_care_visit_detail:
            self._set_omop_variables_from_domain("visit_detail")
            self._c_start_datetime = self._table.c.visit_detail_start_datetime
            self._c_end_datetime = self._table.c.visit_detail_end_datetime
            self._c_type_concept_id = self._table.c.visit_detail_type_concept_id
        else:
            self._set_omop_variables_from_domain("visit")
            self._c_start_datetime = self._table.c.visit_start_datetime
            self._c_end_datetime = self._table.c.visit_end_datetime
            self._c_type_concept_id = self._table.c.visit_type_concept_id

        self._concept = Concept(
            concept_id=OMOPConcepts.VISIT_TYPE_STILL_PATIENT.value,
            concept_name="Still patient",
            concept_code="30",
            domain_id="Visit",
            vocabulary_id="UB04 Pt dis status",
            concept_class_id="UB04 Pt dis status",
        )

    def _sql_header(
        self, distinct_person: bool = True, person_id: int | None = None
    ) -> Select:
        """
        Get the SQL header for the criterion.
        """

        query = select(
            self._table.c.person_id,
            self._c_start_datetime.label("interval_start"),
            self._c_end_datetime.label("interval_end"),
            column_interval_type(IntervalType.POSITIVE),
        ).select_from(self._table)

        return query

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        query = self._sql_header()

        query = query.filter(
            self._c_type_concept_id == OMOPConcepts.VISIT_TYPE_STILL_PATIENT.value
        )

        query = self._filter_datetime(query)

        return query

    def description(self) -> str:
        """
        Get a human-readable description of the criterion.
        """
        return f"{self.__class__.__name__}[]"

    def dict(self) -> dict[str, Any]:
        """
        Get a JSON representation of the criterion.
        """
        return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivePatients":
        """
        Create a criterion from a JSON representation.
        """
        return cls()


class PatientsActiveDuringPeriod(ActivePatients):
    """
    Select Patients who were hospitalized during a given period
    """

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        query = self._sql_header()

        query = self._filter_datetime(query)

        return query
