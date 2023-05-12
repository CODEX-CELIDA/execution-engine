import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from tests.execution_engine.omop.criterion.test_occurrence import Occurrence
from tests.functions import create_condition


class TestConditionOccurrence(Occurrence):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=37311061,
            concept_name="COVID-19",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            concept_code="840539006",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return ConditionOccurrence

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        return create_condition(
            vo=visit_occurrence,
            condition_concept_id=concept_id,
            condition_start_datetime=start_datetime,
            condition_end_datetime=end_datetime,
        )
