import datetime

import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.combination.temporal import (
    TemporalIndicatorCombination,
    TimeIntervalType,
)
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.task.process import get_processing_module
from execution_engine.util.types import Dosage, TimeRange
from execution_engine.util.value import ValueNumber
from tests._fixtures.concept import (
    concept_artificial_respiration,
    concept_body_weight,
    concept_covid19,
    concept_heparin_ingredient,
    concept_unit_kg,
    concept_unit_mg,
)
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
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


class TestTemporalIndicatorCombination:
    """
    Test class for testing criterion combinations (without database).
    """

    @pytest.fixture
    def mock_criteria(self):
        return [MockCriterion(f"c{i}") for i in range(1, 6)]

    def test_criterion_combination_init(self, mock_criteria):
        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_LEAST, threshold=1
        )
        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            start_time=datetime.time(8, 0),
            end_time=datetime.time(16, 0),
        )

        assert combination.operator == operator
        assert len(combination) == 0

    def test_criterion_combination_add(self, mock_criteria):
        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_LEAST, threshold=1
        )
        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            start_time=datetime.time(8, 0),
            end_time=datetime.time(16, 0),
        )

        for criterion in mock_criteria:
            combination.add(criterion)

        assert len(combination) == len(mock_criteria)

        for idx, criterion in enumerate(combination):
            assert criterion == mock_criteria[idx]

    def test_criterion_combination_dict(self, mock_criteria):
        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_LEAST, threshold=1
        )
        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            start_time=datetime.time(8, 0),
            end_time=datetime.time(16, 0),
        )

        for criterion in mock_criteria:
            combination.add(criterion)

        combination_dict = combination.dict()
        assert combination_dict == {
            "operator": "AT_LEAST",
            "threshold": 1,
            "category": "POPULATION_INTERVENTION",
            "start_time": "08:00:00",
            "end_time": "16:00:00",
            "interval_type": None,
            "criteria": [
                {"class_name": "MockCriterion", "data": criterion.dict()}
                for criterion in mock_criteria
            ],
        }

    def test_criterion_combination_from_dict(self, mock_criteria):
        # Register the mock criterion class
        from execution_engine.omop.criterion import factory

        factory.register_criterion_class("MockCriterion", MockCriterion)

        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_LEAST, threshold=1
        )

        combination_data = {
            "operator": "AT_LEAST",
            "threshold": 1,
            "category": "POPULATION_INTERVENTION",
            "start_time": "08:00:00",
            "end_time": "16:00:00",
            "interval_type": None,
            "criteria": [
                {"class_name": "MockCriterion", "data": criterion.dict()}
                for criterion in mock_criteria
            ],
        }

        combination = TemporalIndicatorCombination.from_dict(combination_data)

        assert combination.operator == operator
        assert len(combination) == len(mock_criteria)
        assert combination.start_time == datetime.time(8, 0)
        assert combination.end_time == datetime.time(16, 0)
        assert combination.interval_type is None

        for idx, criterion in enumerate(combination):
            assert str(criterion) == str(mock_criteria[idx])

        combination_data = {
            "operator": "AT_LEAST",
            "threshold": 1,
            "category": "POPULATION_INTERVENTION",
            "start_time": None,
            "end_time": None,
            "interval_type": TimeIntervalType.MORNING_SHIFT,
            "criteria": [
                {"class_name": "MockCriterion", "data": criterion.dict()}
                for criterion in mock_criteria
            ],
        }

        combination = TemporalIndicatorCombination.from_dict(combination_data)

        assert combination.operator == operator
        assert len(combination) == len(mock_criteria)
        assert combination.start_time is None
        assert combination.end_time is None
        assert combination.interval_type == TimeIntervalType.MORNING_SHIFT

        for idx, criterion in enumerate(combination):
            assert str(criterion) == str(mock_criteria[idx])

    @pytest.mark.parametrize("operator", ["AT_LEAST", "AT_MOST", "EXACTLY"])
    def test_operator_with_threshold(self, operator):
        with pytest.raises(
            AssertionError, match=f"Threshold must be set for operator {operator}"
        ):
            TemporalIndicatorCombination.Operator(operator)

    def test_operator(self):
        with pytest.raises(AssertionError, match=""):
            TemporalIndicatorCombination.Operator("invalid")

    @pytest.mark.parametrize(
        "operator, threshold",
        [("AT_LEAST", 1), ("AT_MOST", 1), ("EXACTLY", 1)],
    )
    def test_operator_str(self, operator, threshold):
        op = TemporalIndicatorCombination.Operator(operator, threshold)

        if operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
            assert repr(op) == f'Operator("{operator}", threshold={threshold})'
            assert str(op) == f"{operator}(threshold={threshold})"
        else:
            assert repr(op) == f'Operator("{operator}")'
            assert str(op) == f"{operator}"

    def test_repr(self):
        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_LEAST, threshold=1
        )
        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            interval_type=TimeIntervalType.MORNING_SHIFT,
        )

        assert (
            repr(combination)
            == "TemporalIndicatorCombination(AT_LEAST(threshold=1)).POPULATION_INTERVENTION for morning_shift"
        )

        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            start_time=datetime.time(8, 0),
            end_time=datetime.time(16, 0),
        )

        assert (
            repr(combination)
            == "TemporalIndicatorCombination(AT_LEAST(threshold=1)).POPULATION_INTERVENTION from 08:00:00 to 16:00:00"
        )

    def test_add_all(self):
        operator = TemporalIndicatorCombination.Operator(
            TemporalIndicatorCombination.Operator.AT_MOST, threshold=1
        )
        combination = TemporalIndicatorCombination(
            operator=operator,
            category=CohortCategory.POPULATION_INTERVENTION,
            interval_type=TimeIntervalType.MORNING_SHIFT,
        )

        assert len(combination) == 0

        combination.add_all([MockCriterion("c1"), MockCriterion("c2")])

        assert len(combination) == 2

        assert str(combination[0]) == str(MockCriterion("c1"))
        assert str(combination[1]) == str(MockCriterion("c2"))


c1 = DrugExposure(
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
    category=CohortCategory.POPULATION,
    concept=concept_covid19,
)

c3 = ProcedureOccurrence(
    category=CohortCategory.POPULATION,
    concept=concept_artificial_respiration,
)

bodyweight_measurement_without_forward_fill = Measurement(
    category=CohortCategory.POPULATION,
    concept=concept_body_weight,
    value=ValueNumber.parse("<=110", unit=concept_unit_kg),
    static=False,
    forward_fill=False,
)

bodyweight_measurement_with_forward_fill = Measurement(
    category=CohortCategory.POPULATION,
    concept=concept_body_weight,
    value=ValueNumber.parse("<=110", unit=concept_unit_kg),
    static=False,
)


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
        c1.id = 1
        c2.id = 2
        c3.id = 3
        bodyweight_measurement_without_forward_fill.id = 4
        bodyweight_measurement_with_forward_fill.id = 5

        self.register_criterion(c1, db_session)
        self.register_criterion(c2, db_session)
        self.register_criterion(c3, db_session)
        self.register_criterion(bodyweight_measurement_without_forward_fill, db_session)
        self.register_criterion(bodyweight_measurement_with_forward_fill, db_session)

        return [
            c1,
            c2,
            c3,
            bodyweight_measurement_without_forward_fill,
            bodyweight_measurement_with_forward_fill,
        ]

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

        self.insert_criterion_combination(
            db_session, combination, base_criterion, observation_window
        )

        df = self.fetch_interval_result(
            db_session,
            pi_pair_id=self.pi_pair_id,
            criterion_id=None,
            category=CohortCategory.POPULATION,
        )

        df = df.query("interval_type == 'POSITIVE'")

        for person in persons:
            result = df.query(f"person_id=={person.person_id}")
            result_tuples = set(
                result[["interval_start", "interval_end"]].itertuples(
                    index=False, name=None
                )
            )

            assert result_tuples == expected[person.person_id]


class TestTemporalIndicatorCombinationResultShortObservationWindow(
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

        # return TimeRange(
        #     name="visit", start="2023-03-01 09:36:24Z", end="2023-03-31 14:21:11Z"
        # )

    @pytest.mark.parametrize(
        "combination,expected",
        [
            ####################
            # Full Day
            ####################
            (
                TemporalIndicatorCombination.Day(c1, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 10:36:24+01:00"),
                            pendulum.parse("2023-03-02 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Day(c2, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 00:00:00+01:00"),
                            pendulum.parse("2023-03-03 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Day(c3, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 00:00:00+01:00"),
                            pendulum.parse("2023-03-02 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            ####################
            # Explicit Times
            ####################
            (
                TemporalIndicatorCombination.Presence(
                    c1,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(16, 59),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 10:36:24+01:00"),
                            pendulum.parse("2023-03-01 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 08:30:00+01:00"),
                            pendulum.parse("2023-03-02 16:59:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c1,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(18, 59),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 10:36:24+01:00"),
                            pendulum.parse("2023-03-01 18:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 08:30:00+01:00"),
                            pendulum.parse("2023-03-02 18:59:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c2,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(16, 59),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-03 08:30:00+01:00"),
                            pendulum.parse("2023-03-03 16:59:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c3,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(16, 59),
                    category=CohortCategory.POPULATION,
                ),
                {1: set(), 2: set(), 3: set()},
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c3,
                    start_time=datetime.time(17, 30),
                    end_time=datetime.time(22, 00),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 17:30:00+01:00"),
                            pendulum.parse("2023-03-02 22:00:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            ####################
            # Morning Shifts
            ####################
            (
                TemporalIndicatorCombination.MorningShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 10:36:24+01:00"),
                            pendulum.parse("2023-03-01 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 06:00:00+01:00"),
                            pendulum.parse("2023-03-02 13:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.MorningShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-03 06:00:00+01:00"),
                            pendulum.parse("2023-03-03 13:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.MorningShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {1: set(), 2: set(), 3: set()},
            ),
            ####################
            # Afternoon Shifts
            ####################
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 14:00:00+01:00"),
                            pendulum.parse("2023-03-01 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 14:00:00+01:00"),
                            pendulum.parse("2023-03-02 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 14:00:00+01:00"),
                            pendulum.parse("2023-03-02 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 14:00:00+01:00"),
                            pendulum.parse("2023-03-03 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 14:00:00+01:00"),
                            pendulum.parse("2023-03-02 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            ####################
            # Night Shifts
            ####################
            (
                TemporalIndicatorCombination.NightShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 22:00:00+01:00"),
                            pendulum.parse("2023-03-02 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 22:00:00+01:00"),
                            pendulum.parse("2023-03-03 05:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.NightShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-02 22:00:00+01:00"),
                            pendulum.parse("2023-03-03 05:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.NightShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {1: set(), 2: set(), 3: set()},
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


class TestCriterionCombinationResultLongObservationWindow(
    TestCriterionCombinationDatabase
):
    """
    Test class for testing criterion combinations on the database with a long observation window.

    This class tests only Presence
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
            ####################
            # Full Day
            ####################
            (
                TemporalIndicatorCombination.Day(c1, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 00:00:00+01:00"),
                            pendulum.parse("2023-03-09 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Day(c2, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-03 00:00:00+01:00"),
                            pendulum.parse("2023-03-06 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Day(c3, CohortCategory.POPULATION),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-05 00:00:00+01:00"),
                            pendulum.parse("2023-03-08 23:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            ####################
            # Explicit Times
            ####################
            (
                TemporalIndicatorCombination.Presence(
                    c1,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(16, 59),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 08:30:00+01:00"),
                            pendulum.parse("2023-03-01 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 08:30:00+01:00"),
                            pendulum.parse("2023-03-02 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 08:30:00+01:00"),
                            pendulum.parse("2023-03-03 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 08:30:00+01:00"),
                            pendulum.parse("2023-03-04 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 08:30:00+01:00"),
                            pendulum.parse("2023-03-05 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 08:30:00+01:00"),
                            pendulum.parse("2023-03-06 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 08:30:00+01:00"),
                            pendulum.parse("2023-03-07 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 08:30:00+01:00"),
                            pendulum.parse("2023-03-08 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-09 08:30:00+01:00"),
                            pendulum.parse("2023-03-09 16:59:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c2,
                    start_time=datetime.time(8, 30),
                    end_time=datetime.time(16, 59),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-04 08:30:00+01:00"),
                            pendulum.parse("2023-03-04 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 08:30:00+01:00"),
                            pendulum.parse("2023-03-05 16:59:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 08:30:00+01:00"),
                            pendulum.parse("2023-03-06 16:59:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.Presence(
                    c3,
                    start_time=datetime.time(17, 30),
                    end_time=datetime.time(22, 00),
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-05 17:30:00+01:00"),
                            pendulum.parse("2023-03-05 22:00:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 17:30:00+01:00"),
                            pendulum.parse("2023-03-06 22:00:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 17:30:00+01:00"),
                            pendulum.parse("2023-03-07 22:00:00+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 17:30:00+01:00"),
                            pendulum.parse("2023-03-08 22:00:00+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            # ####################
            # # Morning Shifts
            # ####################
            (
                TemporalIndicatorCombination.MorningShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 06:00:00+01:00"),
                            pendulum.parse("2023-03-01 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 06:00:00+01:00"),
                            pendulum.parse("2023-03-02 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 06:00:00+01:00"),
                            pendulum.parse("2023-03-03 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 06:00:00+01:00"),
                            pendulum.parse("2023-03-04 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 06:00:00+01:00"),
                            pendulum.parse("2023-03-05 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 06:00:00+01:00"),
                            pendulum.parse("2023-03-06 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 06:00:00+01:00"),
                            pendulum.parse("2023-03-07 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 06:00:00+01:00"),
                            pendulum.parse("2023-03-08 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-09 06:00:00+01:00"),
                            pendulum.parse("2023-03-09 13:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.MorningShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-04 06:00:00+01:00"),
                            pendulum.parse("2023-03-04 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 06:00:00+01:00"),
                            pendulum.parse("2023-03-05 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 06:00:00+01:00"),
                            pendulum.parse("2023-03-06 13:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.MorningShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-05 06:00:00+01:00"),
                            pendulum.parse("2023-03-05 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 06:00:00+01:00"),
                            pendulum.parse("2023-03-06 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 06:00:00+01:00"),
                            pendulum.parse("2023-03-07 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 06:00:00+01:00"),
                            pendulum.parse("2023-03-08 13:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            # ####################
            # # Afternoon Shifts
            # ####################
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 14:00:00+01:00"),
                            pendulum.parse("2023-03-01 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 14:00:00+01:00"),
                            pendulum.parse("2023-03-02 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 14:00:00+01:00"),
                            pendulum.parse("2023-03-03 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 14:00:00+01:00"),
                            pendulum.parse("2023-03-04 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 14:00:00+01:00"),
                            pendulum.parse("2023-03-05 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 14:00:00+01:00"),
                            pendulum.parse("2023-03-06 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 14:00:00+01:00"),
                            pendulum.parse("2023-03-07 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 14:00:00+01:00"),
                            pendulum.parse("2023-03-08 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-09 14:00:00+01:00"),
                            pendulum.parse("2023-03-09 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-03 14:00:00+01:00"),
                            pendulum.parse("2023-03-03 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 14:00:00+01:00"),
                            pendulum.parse("2023-03-04 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 14:00:00+01:00"),
                            pendulum.parse("2023-03-05 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 14:00:00+01:00"),
                            pendulum.parse("2023-03-06 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-05 14:00:00+01:00"),
                            pendulum.parse("2023-03-05 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 14:00:00+01:00"),
                            pendulum.parse("2023-03-06 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 14:00:00+01:00"),
                            pendulum.parse("2023-03-07 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 14:00:00+01:00"),
                            pendulum.parse("2023-03-08 21:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            # ####################
            # # Night Shifts
            # ####################
            (
                TemporalIndicatorCombination.NightShift(
                    c1, category=CohortCategory.POPULATION
                ),
                {
                    1: {  # note: drug event is a full day event (despite the drug administration starting only at 18)
                        (
                            pendulum.parse("2023-02-28 22:00:00+01:00"),
                            pendulum.parse("2023-03-01 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-01 22:00:00+01:00"),
                            pendulum.parse("2023-03-02 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 22:00:00+01:00"),
                            pendulum.parse("2023-03-03 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 22:00:00+01:00"),
                            pendulum.parse("2023-03-04 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 22:00:00+01:00"),
                            pendulum.parse("2023-03-05 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 22:00:00+01:00"),
                            pendulum.parse("2023-03-06 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 22:00:00+01:00"),
                            pendulum.parse("2023-03-07 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 22:00:00+01:00"),
                            pendulum.parse("2023-03-08 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-08 22:00:00+01:00"),
                            pendulum.parse("2023-03-09 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-09 22:00:00+01:00"),
                            pendulum.parse("2023-03-10 05:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.NightShift(
                    c2, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-03 22:00:00+01:00"),
                            pendulum.parse("2023-03-04 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-04 22:00:00+01:00"),
                            pendulum.parse("2023-03-05 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 22:00:00+01:00"),
                            pendulum.parse("2023-03-06 05:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
            (
                TemporalIndicatorCombination.NightShift(
                    c3, category=CohortCategory.POPULATION
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-04 22:00:00+01:00"),
                            pendulum.parse("2023-03-05 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-05 22:00:00+01:00"),
                            pendulum.parse("2023-03-06 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-06 22:00:00+01:00"),
                            pendulum.parse("2023-03-07 05:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-07 22:00:00+01:00"),
                            pendulum.parse("2023-03-08 05:59:59+01:00"),
                        ),
                    },
                    2: set(),
                    3: set(),
                },
            ),
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


class TestCriterionPointInTime(TestCriterionCombinationDatabase):
    """
    Test class for testing the behavior of PointInTimeCriterion
    classes.

    More precisely, the test ensures that point-in-time events like
    measurements interact correctly with PointInTimeCriteria and
    TemporalIndicatorCombinations. A particular failure mode of this
    combination has been that single point-in-time event could lead to
    POSITIVE result interval on subsequent days due to forward_fill
    logic in PointInTimeCriterion.
    """

    @pytest.fixture
    def observation_window(self) -> TimeRange:
        return TimeRange(
            start="2023-02-28 13:55:00Z",
            end="2023-03-03 18:00:00Z",
            name="observation",
        )

    def patient_events(self, db_session, visit_occurrence):
        e1 = create_measurement(
            vo=visit_occurrence,
            measurement_concept_id=concept_body_weight.concept_id,
            measurement_datetime=pendulum.parse("2023-03-01 09:00:00+01:00"),
            value_as_number=100,
            unit_concept_id=concept_unit_kg.concept_id,
        )
        db_session.add_all([e1])
        db_session.commit()

    @pytest.mark.parametrize(
        "combination,expected",
        [
            (
                TemporalIndicatorCombination.MorningShift(
                    bodyweight_measurement_without_forward_fill,
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 06:00:00+01:00"),
                            pendulum.parse("2023-03-01 13:59:59+01:00"),
                        ),
                    }
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    bodyweight_measurement_without_forward_fill,
                    category=CohortCategory.POPULATION,
                ),
                {1: set()},
            ),
            (
                TemporalIndicatorCombination.MorningShift(
                    bodyweight_measurement_with_forward_fill,
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 06:00:00+01:00"),
                            pendulum.parse("2023-03-01 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 06:00:00+01:00"),
                            pendulum.parse("2023-03-02 13:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 06:00:00+01:00"),
                            pendulum.parse("2023-03-03 13:59:59+01:00"),
                        ),
                    }
                },
            ),
            (
                TemporalIndicatorCombination.AfternoonShift(
                    bodyweight_measurement_with_forward_fill,
                    category=CohortCategory.POPULATION,
                ),
                {
                    1: {
                        (
                            pendulum.parse("2023-03-01 14:00:00+01:00"),
                            pendulum.parse("2023-03-01 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-02 14:00:00+01:00"),
                            pendulum.parse("2023-03-02 21:59:59+01:00"),
                        ),
                        (
                            pendulum.parse("2023-03-03 14:00:00+01:00"),
                            pendulum.parse("2023-03-03 15:00:00+00:00"),
                        ),
                    }
                },
            ),
        ],
    )
    def test_point_in_time_criterion_on_database(
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
