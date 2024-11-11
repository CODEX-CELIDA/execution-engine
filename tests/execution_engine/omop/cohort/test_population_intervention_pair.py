from execution_engine.omop.cohort.population_intervention_pair import (
    PopulationInterventionPair,
)
from tests.mocks.criterion import MockCriterion


class TestPopulationInterventionPair:

    def test_serialization(self):
        # Register the mock criterion class
        from execution_engine.omop.criterion import factory

        factory.register_criterion_class("MockCriterion", MockCriterion)

        original = PopulationInterventionPair(
            name="foo", url="bar", base_criterion=MockCriterion("c")
        )

        json = original.json()
        deserialized = PopulationInterventionPair.from_json(json)
        assert original == deserialized
