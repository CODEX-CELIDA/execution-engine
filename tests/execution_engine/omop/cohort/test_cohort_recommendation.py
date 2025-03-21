import pytest

from execution_engine.omop.cohort import PopulationInterventionPairExpr
from execution_engine.omop.cohort.recommendation import Recommendation
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_occurrence import ActivePatients
from execution_engine.util import logic
from tests.mocks.criterion import MockCriterion


class TestRecommendation:

    def test_serialization(self):
        # Register the mock criterion class

        original = Recommendation(
            expr=logic.BooleanFunction(),
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
            expr=logic.BooleanFunction(),
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

    def test_database_serialization(
        self,
        concept,
        db_session,  # use_db_session fixture for a proper database clean up after this test
    ):
        from execution_engine.execution_engine import ExecutionEngine

        e = ExecutionEngine(verbose=False)

        url = "http://example.com/fhir/recommendation"

        base_criterion = MockCriterion("c")
        population_criterion = MockCriterion("p")
        intervention_criterion = MockCriterion("i)")

        pi_pair = PopulationInterventionPairExpr(
            population_expr=population_criterion,
            intervention_expr=intervention_criterion,
            name="foo",
            url="foo",
            base_criterion=base_criterion,
        )

        pi_pairs = [pi_pair]

        recommendation = Recommendation(
            expr=pi_pair,
            base_criterion=ActivePatients(),
            name="foo",
            title="bar",
            url=url,
            version="1.0",
            description="test",
            package_version="1.0",
        )

        with pytest.raises(ValueError, match=r"Database ID has not been set yet!"):
            assert recommendation.id is None

        with pytest.raises(ValueError, match=r"Database ID has not been set yet!"):
            assert recommendation.base_criterion.id is None

        for pi_pair in recommendation.population_intervention_pairs():
            with pytest.raises(ValueError, match=r"Database ID has not been set yet!"):
                assert pi_pair.id is None

        for criterion in recommendation.atoms():
            with pytest.raises(ValueError, match=r"Database ID has not been set yet!"):
                assert criterion.id is None

        e.register_recommendation(recommendation)

        assert recommendation.id is not None
        assert recommendation.base_criterion.id is not None

        for pi_pair in recommendation.population_intervention_pairs():
            assert pi_pair.id is not None

        for criterion in recommendation.atoms():
            assert criterion.id is not None

        rec_loaded = e.load_recommendation_from_database(url)

        assert recommendation == rec_loaded
        assert rec_loaded.id == recommendation.id
        assert len(list(rec_loaded.population_intervention_pairs())) == 1

        for pi_pair_loaded, pi_pair in zip(
            list(rec_loaded.population_intervention_pairs()), pi_pairs
        ):
            assert pi_pair_loaded.id == pi_pair.id

        for criterion_loaded, criterion in zip(
            rec_loaded.atoms(), recommendation.atoms()
        ):
            assert criterion.id == criterion_loaded.id
