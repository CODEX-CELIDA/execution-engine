import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.combination import CriterionCombination
from tests._fixtures.mock import MockCriterion


class TestCriterionCombination:
    @pytest.fixture
    def mock_criteria(self):
        return [MockCriterion(f"c{i}") for i in range(1, 6)]

    def test_criterion_combination_init(self, mock_criteria):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination = CriterionCombination(
            "combination",
            exclude=False,
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        assert (
            combination.name
            == "CriterionCombination(AND).population_intervention.combination(exclude=False)"
        )
        assert combination.operator == operator
        assert len(combination) == 0

    def test_criterion_combination_add(self, mock_criteria):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination = CriterionCombination(
            "combination",
            exclude=False,
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        for criterion in mock_criteria:
            combination.add(criterion)

        assert len(combination) == len(mock_criteria)

        for idx, criterion in enumerate(combination):
            assert criterion == mock_criteria[idx]

    def test_criterion_combination_dict(self, mock_criteria):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination = CriterionCombination(
            "combination",
            exclude=False,
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        for criterion in mock_criteria:
            combination.add(criterion)

        combination_dict = combination.dict()
        assert combination_dict == {
            "name": "combination",
            "exclude": False,
            "operator": "AND",
            "threshold": None,
            "category": "population_intervention",
            "criteria": [
                {"class_name": "MockCriterion", "data": criterion.dict()}
                for criterion in mock_criteria
            ],
        }

    def test_criterion_combination_from_dict(self, mock_criteria):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination_data = {
            "name": "combination",
            "exclude": False,
            "operator": "AND",
            "threshold": None,
            "category": "population_intervention",
            "criteria": [
                {"class_name": "MockCriterion", "data": criterion.dict()}
                for criterion in mock_criteria
            ],
        }

        # Register the mock criterion class
        from execution_engine.omop.criterion import factory

        factory.register_criterion_class("MockCriterion", MockCriterion)

        combination = CriterionCombination.from_dict(combination_data)

        assert (
            combination.name
            == "CriterionCombination(AND).population_intervention.combination(exclude=False)"
        )
        assert combination.operator == operator
        assert len(combination) == len(mock_criteria)

        for idx, criterion in enumerate(combination):
            assert str(criterion) == str(mock_criteria[idx])

    @pytest.mark.parametrize("operator", ["AT_LEAST", "AT_MOST", "EXACTLY"])
    def test_operator_with_threshold(self, operator):
        with pytest.raises(
            AssertionError, match=f"Threshold must be set for operator {operator}"
        ):
            CriterionCombination.Operator(operator)

    def test_operator(self):
        with pytest.raises(AssertionError, match=""):
            CriterionCombination.Operator("invalid")

    @pytest.mark.parametrize(
        "operator, threshold",
        [("AND", None), ("OR", None), ("AT_LEAST", 1), ("AT_MOST", 1), ("EXACTLY", 1)],
    )
    def test_operator_str(self, operator, threshold):
        op = CriterionCombination.Operator(operator, threshold)

        if operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
            assert str(op) == f"Operator(operator={operator}, threshold={threshold})"
        else:
            assert str(op) == f"Operator(operator={operator})"

    def test_repr(self):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination = CriterionCombination(
            "combination",
            exclude=False,
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        assert (
            repr(combination)
            == "CriterionCombination(AND).population_intervention.combination(exclude=False)"
        )

    def test_add_all(self):
        operator = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        combination = CriterionCombination(
            "combination",
            exclude=False,
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        assert len(combination) == 0

        combination.add_all([MockCriterion("c1"), MockCriterion("c2")])

        assert len(combination) == 2

        assert str(combination[0]) == str(MockCriterion("c1"))
        assert str(combination[1]) == str(MockCriterion("c2"))
