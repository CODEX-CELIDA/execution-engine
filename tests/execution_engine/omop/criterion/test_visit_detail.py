import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_detail import VisitDetail
from tests.execution_engine.omop.criterion.test_occurrence_criterion import Occurrence
from tests.functions import create_visit_detail


class TestVisitDetail(Occurrence):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=32037,
            concept_name="Intensive Care",
            domain_id="Visit",
            vocabulary_id="Visit",
            concept_class_id="Visit",
            standard_concept="S",
            concept_code="OMOP4822460",
            invalid_reason=None,
        )

    @pytest.fixture
    def concept_no_match(self):
        return Concept(
            concept_id=581478,
            concept_name="Ambulance Visit",
            domain_id="Visit",
            vocabulary_id="Visit",
            concept_class_id="Visit",
            standard_concept="S",
            concept_code="OMOP4822457",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return VisitDetail

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        return create_visit_detail(
            vo=visit_occurrence,
            visit_detail_concept_id=concept_id,
            visit_detail_start_datetime=start_datetime,
            visit_detail_end_datetime=end_datetime,
        )
