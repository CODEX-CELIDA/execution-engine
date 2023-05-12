import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from tests.execution_engine.omop.criterion.test_occurrence_criterion import Occurrence
from tests.functions import create_procedure


class TestProcedureOccurrence(Occurrence):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=4230167,
            concept_name="Artificial respiration",
            domain_id="Procedure",
            vocabulary_id="SNOMED",
            concept_class_id="Procedure",
            standard_concept="S",
            concept_code="40617009",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return ProcedureOccurrence

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        return create_procedure(
            vo=visit_occurrence,
            procedure_concept_id=concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
