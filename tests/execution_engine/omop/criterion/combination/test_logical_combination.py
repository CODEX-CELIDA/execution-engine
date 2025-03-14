import datetime

import pandas as pd
import pendulum
import pytest
import sympy

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.noop import NoopCriterion
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.task.process import get_processing_module
from execution_engine.util import logic
from execution_engine.util.types import Dosage, TimeRange
from execution_engine.util.value import ValueNumber
from tests._fixtures.concept import (
    concept_artificial_respiration,
    concept_body_weight,
    concept_covid19,
    concept_heparin_ingredient,
    concept_tidal_volume,
    concept_unit_kg,
    concept_unit_mg,
    concept_unit_ml,
)
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
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


class TestExpr:
    """
    Test class for testing Expr
    """

    @pytest.fixture
    def mock_criteria(self):
        return [MockCriterion(f"c{i}") for i in range(1, 6)]

    def test_expr_dict(self, mock_criteria):

        expr = logic.And(*mock_criteria)

        combination_dict = expr.dict()

        assert combination_dict == {
            "type": "And",
            "data": {"args": [criterion.dict() for criterion in mock_criteria]},
        }

    def test_expr_from_dict(self, mock_criteria):
        expr_data = {
            "type": "And",
            "data": {"args": [criterion.dict() for criterion in mock_criteria]},
        }

        expr = logic.Expr.from_dict(expr_data)

        assert len(expr.args) == len(mock_criteria)

        for idx, criterion in enumerate(expr.args):
            assert str(criterion) == str(mock_criteria[idx])

    @pytest.mark.parametrize(
        "expr_class",
        [
            logic.And,
            lambda *args: logic.MinCount(*args, threshold=10),
        ],
    )
    def test_criterion_combination_serialization(self, expr_class, mock_criteria):
        # Register the mock criterion class

        expr = expr_class(*mock_criteria)

        json = expr.json()
        deserialized = logic.Expr.from_json(json)

        assert expr == deserialized

    def test_noncommutative_logical_criterion_combination_serialization(
        self, mock_criteria
    ):
        expr = logic.ConditionalFilter(
            left=mock_criteria[0],
            right=mock_criteria[1],
        )

        json = expr.json()
        deserialized = logic.Expr.from_json(json)

        assert expr == deserialized

    @pytest.mark.skip(
        reason="repr does not have a fixed argument order, therefore test fails randomly"
    )
    def test_repr(self, mock_criteria):
        expr = logic.And(*mock_criteria)

        assert repr(expr) == ("LogicalCriterionCombination.And(\n" ")")

    def test_expr_contains_criteria(self, mock_criteria):
        expr = logic.And(*mock_criteria)
        assert len(expr.args) == len(mock_criteria)

        for i in range(len(mock_criteria)):
            assert expr.args[i] == mock_criteria[i]


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
            ingredient_concept=concept_heparin_ingredient,
            dose=Dosage(
                dose=ValueNumber(value_min=15, unit=concept_unit_mg),
                frequency=1,
                interval="d",
            ),
            route=None,
        )

        c2 = ConditionOccurrence(
            concept=concept_covid19,
        )

        c3 = ProcedureOccurrence(
            concept=concept_artificial_respiration,
        )

        c1.set_id(1)
        c2.set_id(2)
        c3.set_id(3)

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
            cls = logic.And
        elif c.func == sympy.Or:
            cls = logic.Or
        elif isinstance(c.func, sympy.core.function.UndefinedFunction):
            if c.func.name in ["MinCount", "MaxCount", "ExactCount"]:
                assert args[0].is_number, "First argument must be a number (threshold)"
                threshold = int(args[0])
                args = args[1:]
            if c.func.name == "MinCount":
                cls = lambda *args: logic.MinCount(*args, threshold=threshold)
            elif c.func.name == "MaxCount":
                cls = lambda *args: logic.MaxCount(*args, threshold=threshold)
            elif c.func.name == "ExactCount":
                cls = lambda *args: logic.ExactCount(*args, threshold=threshold)
            elif c.func.name == "AllOrNone":
                cls = lambda *args: logic.AllOrNone(*args)
            elif c.func.name == "ConditionalFilter":
                cls = lambda *args: logic.ConditionalFilter(*args)
            else:
                raise ValueError(f"Unknown operator {c.func}")
        else:
            raise ValueError(f"Unknown operator {c.func}")

        # c1, c2, c3 = [
        #     c for c in criteria
        # ]  # TODO(jmoringe): copy should no longer be necessary

        c1, c2, c3 = criteria

        for arg in args:
            if arg.is_Not:
                if arg.args[0].name == "c1":
                    c1 = logic.Not(c1)
                elif arg.args[0].name == "c2":
                    c2 = logic.Not(c2)
                elif arg.args[0].name == "c3":
                    c3 = logic.Not(c3)
                else:
                    raise ValueError(f"Unknown criterion {arg.args[0].name}")

        symbols = {"c1": c1, "c2": c2, "c3": c3}

        if hasattr(c.func, "name") and c.func.name == "ConditionalFilter":
            assert len(c.args) == 2

            comb = logic.ConditionalFilter(
                left=symbols[str(c.args[0])],
                right=symbols[str(c.args[1])],
            )

        else:
            comb = cls(
                *[symbols[str(symbol)] for symbol in c.atoms() if not symbol.is_number]
            )

        if exclude:
            comb = logic.Not(comb)

        noop_criterion = NoopCriterion()
        noop_criterion.set_id(1005)
        noop_intervention = logic.NonSimplifiableAnd(noop_criterion)
        self.register_criterion(noop_criterion, db_session)

        self.insert_expression(
            db_session,
            population=comb,
            intervention=noop_intervention,
            base_criterion=base_criterion,
            observation_window=observation_window,
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


class TestCriterionCombinationNoData(TestCriterionCombinationDatabase):
    """
    Test class for testing criterion combinations on the database.

    The purpose of this class is to test the behavior for a combination like
    NOT(AND(c1, c2)), where c1 returns NO_DATA intervals and c2 returns nothing.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 04:00:00Z", end="2023-03-04 18:00:00Z", name="observation"
        )

    @pytest.fixture
    def criteria(self, db_session):
        c1 = DrugExposure(
            ingredient_concept=concept_heparin_ingredient,
            dose=Dosage(
                dose=ValueNumber(value=100, unit=concept_unit_mg),
                frequency=1,
                interval="d",
            ),
            route=None,
        )

        c2 = Measurement(
            concept=concept_tidal_volume,
            value=ValueNumber(value_min=500, unit=concept_unit_ml),
        )

        c3 = Measurement(
            concept=concept_body_weight,
            value=ValueNumber(value_min=70, unit=concept_unit_kg),
        )

        c1.set_id(1)
        c2.set_id(2)
        c3.set_id(3)

        self.register_criterion(c1, db_session)
        self.register_criterion(c2, db_session)
        self.register_criterion(c3, db_session)

        return [c1, c2, c3]

    @pytest.fixture
    def patient_events(self, db_session, person_visit):
        # no patient events added
        pass

    @pytest.mark.parametrize(
        "combination,expected",
        [
            ("c1 & c2", {1: {}, 2: {}, 3: {}}),
            ("c1 | c2", {1: {}, 2: {}, 3: {}}),
            (
                "~(c1 & c2)",
                {
                    1: {
                        "2023-03-01",
                        "2023-03-02",
                        "2023-03-03",
                        "2023-03-04",
                    },  # admitted on 2023-03-01
                    2: {
                        "2023-03-02",
                        "2023-03-03",
                        "2023-03-04",
                    },  # admitted on 2023-03-02
                    3: {"2023-03-03", "2023-03-04"},  # admitted on 2023-03-03
                },
            ),
            (
                "~(c1 | c2)",  # c1 is NEGATIVE, c2 is NO_DATA -> c1 | c2 is still NO_DATA
                # -> ~(c1 | c2) = ~c1 & ~c2 = (POSITIVE & NO_DATA) = NO_DATA
                #  -> interpreted as "NEGATIVE" in the final result (no POSITIVE intervals there)
                {
                    1: {},
                    2: {},
                    3: {},
                },
            ),
            ("~c1 & c2", {1: {}, 2: {}, 3: {}}),
            (
                "~c1 | c2",  # c1 is NEGATIVE, c2 is NO_DATA -> ~c1 | c2 = POSITIVE | NO_DATA = POSITIVE
                {
                    1: {"2023-03-01", "2023-03-02", "2023-03-03", "2023-03-04"},
                    2: {"2023-03-02", "2023-03-03", "2023-03-04"},
                    3: {"2023-03-03", "2023-03-04"},
                },
            ),
            ("c1 & ~c2", {1: {}, 2: {}, 3: {}}),
            (
                "c1 | ~c2",  # c1 is NEGATIVE, c2 is NO_DATA -> c1 | ~c2 = NEGATIVE | NO_DATA = NO_DATA -> interpreted as "NEGATIVE" in the final result (no POSITIVE intervals there)
                {
                    1: {},
                    2: {},
                    3: {},
                },
            ),
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


class TestCriterionCombinationConditionalFilter(TestCriterionCombinationDatabase):
    """
    Test class for testing criterion combinations on the database.

    The purpose of this class is to test the behavior for a combination like

    NOT(AND(c1, c2)), where c1 returns NO_DATA intervals and c2 returns nothing.

    The result should be a positive interval (because "nothing" is interpreted as NEGATIVE, and ~(NO_DATA & NEGATIVE)
    should be positive. This is a regression test related to bug found when processing body-weight-related drug
    combinations (in rec17) when no body weight is given.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 04:00:00Z", end="2023-03-04 18:00:00Z", name="observation"
        )

    @pytest.fixture
    def criteria(self, db_session):
        c1 = DrugExposure(
            ingredient_concept=concept_heparin_ingredient,
            dose=Dosage(
                dose=ValueNumber(value=100, unit=concept_unit_mg),
                frequency=1,
                interval="d",
            ),
            route=None,
        )

        c2 = Measurement(
            concept=concept_body_weight,
            value=ValueNumber(value_min=70, unit=concept_unit_kg),
        )

        c3 = ConditionOccurrence(
            concept=concept_covid19,
        )

        c1.set_id(1)
        c2.set_id(2)
        c3.set_id(3)

        self.register_criterion(c1, db_session)
        self.register_criterion(c2, db_session)
        self.register_criterion(c3, db_session)

        return [c1, c2, c3]

    @pytest.fixture
    def patient_events(self, db_session, person_visit):
        _, visit_occurrence = person_visit[0]
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e2 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e3 = create_measurement(
            vo=visit_occurrence,
            measurement_concept_id=concept_body_weight.concept_id,
            measurement_datetime=pendulum.parse("2023-03-04 05:00:00+01:00"),
            value_as_number=70,
            unit_concept_id=concept_unit_kg.concept_id,
        )

        db_session.add_all([e1, e2, e3])

        #####################################
        _, visit_occurrence = person_visit[1]
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e2 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        db_session.add_all([e1, e2])

        #####################################
        _, visit_occurrence = person_visit[2]
        e1 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-03 18:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e2 = create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=concept_heparin_ingredient.concept_id,
            start_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            end_datetime=pendulum.parse(
                "2023-03-04 06:00:00+01:00"
            ),  # -> this results in full day positive (because drug exposure with quantity 50 on this day)
            quantity=100,
        )

        e3 = create_condition(
            vo=visit_occurrence,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=pendulum.parse("2023-03-03 18:00:00+01:00"),
            condition_end_datetime=pendulum.parse("2023-03-06 18:00:00+01:00"),
        )

        db_session.add_all([e1, e2, e3])

        db_session.commit()

    @pytest.mark.parametrize(
        "combination,expected",
        [
            (
                "ConditionalFilter(c1, c2)",
                {
                    1: {"2023-03-03", "2023-03-04"},  # admitted on 2023-03-01
                    2: {},  # admitted on 2023-03-02
                    3: {},  # admitted on 2023-03-03
                },
            ),
            (
                "ConditionalFilter(c2, c1)",
                {
                    1: {"2023-03-03", "2023-03-04"},  # admitted on 2023-03-01
                    2: {},  # admitted on 2023-03-02
                    3: {},  # admitted on 2023-03-03
                },
            ),
            (
                "ConditionalFilter(c1, c3)",
                {
                    1: {},  # admitted on 2023-03-01
                    2: {},  # admitted on 2023-03-02
                    3: {"2023-03-04"},  # admitted on 2023-03-03
                },
            ),
            (
                "ConditionalFilter(c3, c1)",
                {
                    1: {},  # admitted on 2023-03-01
                    2: {},  # admitted on 2023-03-02
                    3: {"2023-03-04"},  # admitted on 2023-03-03
                },
            ),
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
