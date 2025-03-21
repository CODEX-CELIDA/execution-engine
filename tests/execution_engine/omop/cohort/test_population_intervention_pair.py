from execution_engine.omop.cohort.population_intervention_pair import (
    PopulationInterventionPairExpr,
)
from execution_engine.util import logic
from tests.mocks.criterion import MockCriterion


class TestPopulationInterventionPair:

    def test_serialization(self):
        # Register the mock criterion class
        original = PopulationInterventionPairExpr(
            population_expr=logic.NonSimplifiableAnd(MockCriterion("population")),
            intervention_expr=logic.NonSimplifiableAnd(MockCriterion("intervention")),
            name="foo",
            url="bar",
            base_criterion=MockCriterion("base"),
        )

        json = original.json()
        deserialized = PopulationInterventionPairExpr.from_json(json)
        assert original == deserialized
