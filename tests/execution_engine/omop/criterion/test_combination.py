import datetime

import pandas as pd
import pendulum
import pytest
import sympy

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.task.process import get_processing_module
from execution_engine.util.types import Dosage, TimeRange
from execution_engine.util.value import ValueNumber
from tests._fixtures.concept import (
    concept_artificial_respiration,
    concept_covid19,
    concept_heparin_ingredient,
    concept_unit_mg,
)
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_procedure,
    create_visit,
)
from tests.functions import intervals_to_df as intervals_to_df_orig
from tests.mocks.criterion import MockCriterion

process = get_processing_module()


def intervals_to_df(result, by=None):
    df = intervals_to_df_orig(result, by, process.normalize_interval)
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert("Europe/Berlin")
    return df


class TestCriterionCombination:
    """
    Test class for testing criterion combinations (without database).
    """

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
            == "CriterionCombination(AND).POPULATION_INTERVENTION.combination(exclude=False)"
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
            "category": "POPULATION_INTERVENTION",
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
            "category": "POPULATION_INTERVENTION",
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
            == "CriterionCombination(AND).POPULATION_INTERVENTION.combination(exclude=False)"
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
            == "CriterionCombination(AND).POPULATION_INTERVENTION.combination(exclude=False)"
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


class TestCriterionCombinationDatabase(TestCriterion):
    """
    Test class for testing criterion combinations on the database.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 04:00:00Z", end="2023-03-04 18:00:00Z", name="observation"
        )

    @pytest.fixture
    def criteria(self, db_session):
        c1 = DrugExposure(
            name="test",
            exclude=False,
            category=CohortCategory.POPULATION,
            ingredient_concept=concept_heparin_ingredient,
            dose=Dosage(
                dose=ValueNumber(value_min=15, unit=concept_unit_mg),
                frequency=1,
                interval="d",
            ),
            route=None,
        )

        c2 = ConditionOccurrence(
            name="test",
            exclude=False,
            category=CohortCategory.POPULATION,
            concept=concept_covid19,
        )

        c3 = ProcedureOccurrence(
            name="test",
            exclude=False,
            category=CohortCategory.POPULATION,
            concept=concept_artificial_respiration,
        )

        c1.id = 1
        c2.id = 2
        c3.id = 3

        self.register_criterion(c1, db_session)
        self.register_criterion(c2, db_session)
        self.register_criterion(c3, db_session)

        return [c1, c2, c3]

    def run_criteria_test(
        self,
        combination,
        expected,
        db_session,
        criteria,
        base_criterion,
        observation_window,
        persons,
    ):
        c = sympy.parse_expr(combination)

        if c.is_Not:
            exclude = True
            c = c.args[0]
        else:
            exclude = False

        args = c.args
        threshold = None

        if c.func == sympy.And:
            operator = CriterionCombination.Operator.AND
        elif c.func == sympy.Or:
            operator = CriterionCombination.Operator.OR
        elif isinstance(c.func, sympy.core.function.UndefinedFunction):
            assert args[0].is_number, "First argument must be a number (threshold)"
            threshold = args[0]
            args = args[1:]
            if c.func.name == "MinCount":
                operator = CriterionCombination.Operator.AT_LEAST
            elif c.func.name == "MaxCount":
                operator = CriterionCombination.Operator.AT_MOST
            elif c.func.name == "ExactCount":
                operator = CriterionCombination.Operator.EXACTLY
            elif c.func.name == "AllOrNone":
                operator = CriterionCombination.Operator.ALL_OR_NONE
            else:
                raise ValueError(f"Unknown operator {c.func}")
        else:
            raise ValueError(f"Unknown operator {c.func}")

        c1, c2, c3 = [c.copy() for c in criteria]

        for arg in args:
            if arg.is_Not:
                if arg.args[0].name == "c1":
                    c1.exclude = True
                elif arg.args[0].name == "c2":
                    c2.exclude = True
                elif arg.args[0].name == "c3":
                    c3.exclude = True
                else:
                    raise ValueError(f"Unknown criterion {arg.args[0].name}")

        comb = CriterionCombination(
            "combination",
            exclude=exclude,
            category=CohortCategory.POPULATION,
            operator=CriterionCombination.Operator(operator, threshold=threshold),
        )

        for symbol in c.atoms():
            if symbol.is_number:
                continue
            elif symbol.name == "c1":
                comb.add(c1)
            elif symbol.name == "c2":
                comb.add(c2)
            elif symbol.name == "c3":
                comb.add(c3)
            else:
                raise ValueError(f"Unknown criterion {symbol.name}")

        self.insert_criterion_combination(
            db_session, comb, base_criterion, observation_window
        )

        df = self.fetch_full_day_result(
            db_session,
            pi_pair_id=self.pi_pair_id,
            criterion_id=None,
            category=CohortCategory.POPULATION,
        )

        for person in persons:
            result = df.query(f"person_id=={person.person_id}")
            expected_result = date_set(expected[person.person_id])

            assert set(pd.to_datetime(result["valid_date"]).dt.date) == expected_result


class TestCriterionCombinationResultShortObservationWindow(
    TestCriterionCombinationDatabase
):
    """
    Test class for testing criterion combinations on the database with a short observation window.

    This class mainly tests the AND, OR and NOT combinations.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 04:00:00Z", end="2023-03-04 18:00:00Z", name="observation"
        )

    @pytest.fixture
    def patient_events(self, db_session, person_visit):
        _, visit_occurrence = person_visit[0]
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-01 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-02 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e2 = create_condition(
            vo=visit_occurrence,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=pendulum.parse("2023-03-02 18:00:00+01:00"),
            condition_end_datetime=pendulum.parse("2023-03-03 18:00:00+01:00"),
        )

        e3 = create_procedure(
            vo=visit_occurrence,
            procedure_concept_id=concept_artificial_respiration.concept_id,
            start_datetime=pendulum.parse("2023-03-02 17:00:00+01:00"),
            end_datetime=pendulum.parse("2023-03-02 18:00:01+01:00"),
        )
        db_session.add_all([e1, e2, e3])
        db_session.commit()

    @pytest.mark.parametrize(
        "combination,expected",
        [
            ("c1 & c2", {1: {}, 2: {}, 3: {}}),
            ("c2 & c3", {1: {}, 2: {}, 3: {}}),
            ("c1 | c2", {1: {"2023-03-01", "2023-03-02"}, 2: {}, 3: {}}),
            ("c1 & c2 & c3", {1: {}, 2: {}, 3: {}}),
            ("c1 | c2 | c3", {1: {"2023-03-01", "2023-03-02"}, 2: {}, 3: {}}),
            ("c1 | c2 & c3", {1: {"2023-03-01", "2023-03-02"}, 2: {}, 3: {}}),
            (
                "~(c1 & c2)",
                {
                    1: {"2023-03-01", "2023-03-03", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            (
                "~(c2 & c3)",
                {
                    1: {"2023-03-01", "2023-03-03", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            (
                "~(c1 | c2)",
                {
                    1: {"2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            (
                "~(c2 | c3)",
                {
                    1: {"2023-03-01", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            (
                "~(c1 | c2 | c3)",
                {
                    1: {"2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            ("~c1 & c2", {1: {}, 2: {}, 3: {}}),
            (
                "~c1 | c2",
                {
                    1: {"2023-03-03", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            ("c1 & ~c2", {1: {"2023-03-01"}, 2: {}, 3: {}}),
            (
                "c1 | ~c2",
                {
                    1: {"2023-03-01", "2023-03-02", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            ("~c2 & c3", {1: {}, 2: {}, 3: {}}),
            (
                "MinCount(1, c1, c2, c3)",
                {1: {"2023-03-01", "2023-03-02"}, 2: {}, 3: {}},
            ),
            ("MaxCount(1, c1, c2, c3)", {1: {"2023-03-01"}, 2: {}, 3: {}}),
            ("ExactCount(1, c1, c2, c3)", {1: {"2023-03-01"}, 2: {}, 3: {}}),
            ("MinCount(2, c1, c2, c3)", {1: {}, 2: {}, 3: {}}),
            ("MaxCount(2, c1, c2, c3)", {1: {"2023-03-01"}, 2: {}, 3: {}}),
            ("ExactCount(2, c1, c2, c3)", {1: {}, 2: {}, 3: {}}),
            ("MinCount(3, c1, c2, c3)", {1: {}, 2: {}, 3: {}}),
            (
                "MaxCount(3, c1, c2, c3)",
                {1: {"2023-03-01", "2023-03-02"}, 2: {}, 3: {}},
            ),
            ("ExactCount(3, c1, c2, c3)", {1: {}, 2: {}, 3: {}}),
        ],
    )
    def test_combination_on_database(
        self,
        person_visit,
        db_session,
        base_criterion,
        patient_events,
        criteria,
        combination,
        expected,
        observation_window,
    ):
        persons = [pv[0] for pv in person_visit]
        self.run_criteria_test(
            combination,
            expected,
            db_session,
            criteria,
            base_criterion,
            observation_window,
            persons,
        )


class TestCriterionCombinationResultLongObservationWindow(
    TestCriterionCombinationDatabase
):
    """
    Test class for testing criterion combinations on the database with a long observation window.

    This class tests only MaxCount, MinCount and ExactCount combinations.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-02-28 13:55:00Z",
            end="2023-03-10 18:00:00Z",
            name="observation",
        )

    def patient_events(self, db_session, visit_occurrence):
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-01 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-09 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=450,
        )

        e2 = create_condition(
            vo=visit_occurrence,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=pendulum.parse("2023-03-03 18:00:00+01:00"),
            condition_end_datetime=pendulum.parse("2023-03-06 18:00:00+01:00"),
        )

        e3 = create_procedure(
            vo=visit_occurrence,
            procedure_concept_id=concept_artificial_respiration.concept_id,
            start_datetime=pendulum.parse("2023-03-05 00:00:00+01:00"),
            end_datetime=pendulum.parse("2023-03-08 18:00:01+01:00"),
        )
        db_session.add_all([e1, e2, e3])
        db_session.commit()

    @pytest.mark.parametrize(
        "combination,expected",
        [
            (
                "MinCount(1, c1, c2, c3)",
                {
                    1: {
                        "2023-03-01",
                        "2023-03-02",
                        "2023-03-03",
                        "2023-03-04",
                        "2023-03-05",
                        "2023-03-06",
                        "2023-03-07",
                        "2023-03-08",
                        "2023-03-09",
                    }
                },
            ),
            (
                "MaxCount(1, c1, c2, c3)",
                {1: {"2023-03-01", "2023-03-02", "2023-03-09"}},
            ),
            (
                "ExactCount(1, c1, c2, c3)",
                {1: {"2023-03-01", "2023-03-02", "2023-03-09"}},
            ),
            (
                "MinCount(2, c1, c2, c3)",
                {
                    1: {
                        "2023-03-04",
                        "2023-03-05",
                        "2023-03-06",
                        "2023-03-07",
                    }
                },
            ),
            (
                "MaxCount(2, c1, c2, c3)",
                {
                    1: {
                        "2023-03-01",
                        "2023-03-02",
                        "2023-03-03",
                        "2023-03-04",
                        "2023-03-07",
                        "2023-03-08",
                        "2023-03-09",
                    }
                },
            ),
            ("ExactCount(2, c1, c2, c3)", {1: {"2023-03-04", "2023-03-07"}}),
            ("MinCount(3, c1, c2, c3)", {1: {"2023-03-05"}}),
            (
                "MaxCount(3, c1, c2, c3)",
                {
                    1: {
                        "2023-03-01",
                        "2023-03-02",
                        "2023-03-03",
                        "2023-03-04",
                        "2023-03-05",
                        "2023-03-06",
                        "2023-03-07",
                        "2023-03-08",
                        "2023-03-09",
                    }
                },
            ),
            ("ExactCount(3, c1, c2, c3)", {1: {"2023-03-05"}}),
        ],
    )
    def test_overlapping_combination_on_database(
        self,
        person,
        db_session,
        base_criterion,
        combination,
        expected,
        observation_window,
        criteria,
    ):
        persons = [person[0]]  # only one person
        vos = [
            create_visit(
                person_id=person.person_id,
                visit_start_datetime=observation_window.start
                + datetime.timedelta(hours=3),
                visit_end_datetime=observation_window.end - datetime.timedelta(hours=3),
                visit_concept_id=concepts.INTENSIVE_CARE,
            )
            for person in persons
        ]

        self.patient_events(db_session, vos[0])

        db_session.add_all(vos)
        db_session.commit()

        self.run_criteria_test(
            combination,
            expected,
            db_session,
            criteria,
            base_criterion,
            observation_window,
            persons,
        )
