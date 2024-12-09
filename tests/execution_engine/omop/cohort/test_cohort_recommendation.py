import pytest

from execution_engine.omop.cohort.recommendation import Recommendation
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_occurrence import ActivePatients
from tests.mocks.criterion import MockCriterion


class TestRecommendation:

    def test_serialization(self):
        # Register the mock criterion class
        from execution_engine.omop.criterion import factory

        factory.register_criterion_class("MockCriterion", MockCriterion)

        original = Recommendation(
            pi_pairs=[],
            base_criterion=MockCriterion("c"),
            name="foo",
            title="bar",
            url="baz",
            version="1.0",
            description="hi",
        )

        json = original.json()
        deserialized = Recommendation.from_json(json)
        assert original == deserialized

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

    def test_serialization_with_active_patients(self, concept):
        # Test with a ActivePatients as the base criterion
        # specifically because there used to be a problem with the
        # serialization of that combination.
        original = Recommendation(
            pi_pairs=[],
            base_criterion=ActivePatients(),
            name="foo",
            title="bar",
            url="baz",
            version="1.0",
            description="hi",
        )

        json = original.json()
        deserialized = Recommendation.from_json(json)
        assert original == deserialized
