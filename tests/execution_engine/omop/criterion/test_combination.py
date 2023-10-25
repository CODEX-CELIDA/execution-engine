import pandas as pd
import pendulum
import pytest
import sympy
from sqlalchemy import func, select, text

from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.cohort_definition import add_result_insert
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.db.celida import RecommendationResult
from execution_engine.util import TimeRange, ValueNumber
from tests._fixtures.concept import (
    concept_covid19,
    concept_heparin_ingredient,
    concept_unit_mg,
)
from tests._fixtures.mock import MockCriterion
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import create_condition, create_drug_exposure


class TestCriterionCombination(TestCriterion):
    @pytest.fixture
    def mock_criteria(self):
        return [MockCriterion(f"c{i}") for i in range(1, 6)]

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 04:00:00", end="2023-03-04 18:00:00", name="observation"
        )

    @pytest.fixture
    def criteria_db(self, db_session, person_visit):
        _, visit_occurrence = person_visit[0]
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse("2023-03-01 18:00:00"),
            end_datetime=pendulum.parse("2023-03-02 06:00:00"),
            quantity=100,
        )

        e2 = create_condition(
            vo=visit_occurrence,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=pendulum.parse("2023-03-02 18:00:00"),
            condition_end_datetime=pendulum.parse("2023-03-03 18:00:00"),
        )
        db_session.add_all([e1, e2])
        db_session.commit()

        c1 = DrugExposure(
            name="test",
            exclude=False,
            category=CohortCategory.POPULATION,
            drug_concepts=[concept_heparin_ingredient.concept_id],
            ingredient_concept=concept_heparin_ingredient,
            dose=ValueNumber(value=50, unit=concept_unit_mg),
            frequency=1,
            interval="d",
            route=None,
        )

        c2 = ConditionOccurrence(
            name="test",
            exclude=False,
            category=CohortCategory.POPULATION,
            concept=concept_covid19,
        )

        c1.id = 1
        c2.id = 2

        return [c1, c2]

    @pytest.mark.parametrize(
        "combination,expected",
        [
            ("c1 & c2", {"2023-03-02"}),
            ("c1 | c2", {"2023-03-01", "2023-03-02", "2023-03-03"}),
            ("~(c1 & c2)", {"2023-03-01", "2023-03-03", "2023-03-04"}),
            ("~(c1 | c2)", {"2023-03-04"}),
            ("~c1 & c2", {"2023-03-03"}),
            ("~c1 | c2", {"2023-03-02", "2023-03-03", "2023-03-04"}),
            ("c1 & ~c2", {"2023-03-01"}),
            ("c1 | ~c2", {"2023-03-01", "2023-03-02", "2023-03-04"}),
        ],
    )
    def test_combination_on_database(
        self,
        person_visit,
        db_session,
        base_table,
        criteria_db,
        combination,
        expected,
        observation_window,
    ):
        # BASE cohort in results table is required for combination to work
        query = (
            select(func.count("*"))
            .select_from(RecommendationResult)
            .where(RecommendationResult.cohort_category == CohortCategory.BASE)
        )
        assert db_session.execute(query).scalar() > 1, "No base cohort in database"

        c = sympy.parse_expr(combination)

        if c.is_Not:
            exclude = True
            c = c.args[0]
        else:
            exclude = False

        if c.func == sympy.And:
            operator = CriterionCombination.Operator.AND
        elif c.func == sympy.Or:
            operator = CriterionCombination.Operator.OR
        else:
            raise ValueError(f"Unknown operator {c.func}")

        c1, c2 = [c.copy() for c in criteria_db]

        for arg in c.args:
            if arg.is_Not:
                if arg.args[0].name == "c1":
                    c1.exclude = True
                elif arg.args[0].name == "c2":
                    c2.exclude = True
                else:
                    raise ValueError(f"Unknown criterion {arg.args[0].name}")

        comb = CriterionCombination(
            "combination",
            exclude=exclude,
            category=CohortCategory.POPULATION_INTERVENTION,
            operator=CriterionCombination.Operator(operator),
        )
        comb.add_all([c1, c2])

        execution_map = ExecutionMap(comb)
        query_params = observation_window.dict() | {"run_id": 1}

        db_session.execute(
            text("SET session_replication_role = 'replica';")
        )  # disable fkey checks (because of recommendation_result.run_id)

        for criterion in execution_map.sequential():
            query = criterion.sql_generate(base_table=base_table)
            query = add_result_insert(
                query,
                plan_id=None,
                criterion_id=criterion.id,
                cohort_category=criterion.category,
            )

            query.description = query.select.description

            db_session.execute(query, params=query_params)

        db_session.commit()
        query = execution_map.combine(CohortCategory.POPULATION_INTERVENTION)

        df = pd.read_sql(query, db_session.connection(), params=query_params)
        df = df.query(f"person_id=={person_visit[0][0].person_id}")

        assert set(pd.to_datetime(df["valid_date"]).dt.date) == date_set(expected)

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
