import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import TimeRange, Timing
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueDuration
from tests.execution_engine.omop.criterion.test_occurrence_criterion import Occurrence
from tests.functions import create_procedure, df_to_datetime_interval, interval


class TestProcedureOccurrence(Occurrence):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=4230167,
            concept_name="Artificial respiration",
            domain_id="Procedure",
            vocabulary_id="SNOMED",
            concept_class_id="Procedure",
            standard_concept="S",
            concept_code="40617009",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return ProcedureOccurrence

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        return create_procedure(
            vo=visit_occurrence,
            procedure_concept_id=concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    def test_timing_duration_no_interval(
        self,
        person_visit,
        db_session,
        concept,
        criterion_execute_func,
        observation_window,
        base_table,
    ):
        _, vo = person_visit[0]

        time_ranges = [
            ("2023-03-04 18:00:00+01:00", "2023-03-04 19:00:00+01:00", False),
            ("2023-03-04 20:00:00+01:00", "2023-03-04 21:30:00+01:00", False),
            ("2023-03-05 19:30:00+01:00", "2023-03-05 21:30:00+01:00", True),
        ]

        def criterion_execute_func_timing(
            concept: Concept,
            exclude: bool,
            value: ValueNumber | None = None,
        ):
            timing = Timing(duration=2 * TimeUnit.HOUR)

            criterion = ProcedureOccurrence(
                exclude=False,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
                timing=timing,
                static=None,
            )
            if exclude:
                criterion = LogicalCriterionCombination.Not(
                    criterion, category=criterion.category
                )
            self.insert_criterion(db_session, criterion, observation_window)
            df = self.fetch_full_day_result(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )

            return df

        self.perform_test(
            person_visit,
            concept,
            db_session,
            criterion_execute_func_timing,
            observation_window,
            time_ranges,
        )

    @pytest.mark.parametrize(
        "time_ranges",
        [
            [
                ("2023-03-04 18:00:00+01:00", "2023-03-04 18:30:00+01:00"),  # 0h30
                ("2023-03-04 20:00:00+01:00", "2023-03-04 21:30:00+01:00"),  # 1h30
                ("2023-03-05 19:30:00+01:00", "2023-03-05 21:30:00+01:00"),  # 2h
            ]
        ],
    )
    @pytest.mark.parametrize(
        "timing,expected_intervals",
        [
            (
                Timing(
                    duration=ValueDuration(value_min=1, unit=TimeUnit.HOUR),
                    frequency=1,
                    interval=1 * TimeUnit.HOUR,
                ),
                {
                    # unmerged intervals
                    # interval("2023-03-04 20:00:00+01:00", "2023-03-04 20:59:59+01:00"),
                    # interval("2023-03-04 21:00:00+01:00", "2023-03-04 21:59:59+01:00"),
                    # interval("2023-03-05 19:00:00+01:00", "2023-03-05 19:59:59+01:00"),
                    # interval("2023-03-05 20:00:00+01:00", "2023-03-05 20:59:59+01:00"),
                    # interval("2023-03-05 21:00:00+01:00", "2023-03-05 21:59:59+01:00"),
                    # merged intervals
                    interval("2023-03-04 20:00:00+01:00", "2023-03-04 21:59:59+01:00"),
                    interval("2023-03-05 19:00:00+01:00", "2023-03-05 21:59:59+01:00"),
                },  # because we are looking for _>=1 hour_ per _1 HOUR_
            ),
            (
                Timing(
                    duration=1 * TimeUnit.HOUR, frequency=1, interval=1 * TimeUnit.HOUR
                ),
                set(),  # because we are looking for _1 hour_ intervals (per 1 HOUR)
            ),
            (
                Timing(
                    duration=1 * TimeUnit.HOUR, frequency=1, interval=1 * TimeUnit.DAY
                ),
                set(),  # because we are looking for _1 hour_ intervals (per day)
            ),
            (
                Timing(
                    duration=ValueDuration(value_min=1, unit=TimeUnit.HOUR),
                    frequency=1,
                    interval=1 * TimeUnit.DAY,
                ),
                {
                    # unmerged intervals
                    # interval("2023-03-04 00:00:00+01:00", "2023-03-04 23:59:59+01:00"),
                    # interval("2023-03-05 00:00:00+01:00", "2023-03-05 23:59:59+01:00"),
                    # merged intervals
                    interval("2023-03-04 00:00:00+01:00", "2023-03-05 23:59:59+01:00"),
                },
            ),
            (
                Timing(
                    duration=2 * TimeUnit.HOUR, frequency=1, interval=1 * TimeUnit.DAY
                ),
                {interval("2023-03-05 00:00:00+01:00", "2023-03-05 23:59:59+01:00")},
            ),
            (
                Timing(
                    duration=2 * TimeUnit.HOUR, frequency=1, interval=2 * TimeUnit.DAY
                ),
                {
                    interval("2023-03-05 00:00:00+01:00", "2023-03-06 23:59:59+01:00"),
                },
                # because we are looking for 2 hour per _2 DAY_, the next day is also correct
            ),
            (
                Timing(
                    duration=2 * TimeUnit.HOUR, frequency=1, interval=2 * TimeUnit.HOUR
                ),
                {
                    # unmerged intervals
                    # interval("2023-03-05 18:00:00+01:00", "2023-03-05 19:59:59+01:00"),
                    # interval("2023-03-05 20:00:00+01:00", "2023-03-05 21:59:59+01:00"),
                    # merged intervals
                    interval("2023-03-05 18:00:00+01:00", "2023-03-05 21:59:59+01:00"),
                },
                # because we are looking for 2 hour per 2 HOUR
            ),
            (
                Timing(
                    duration=ValueDuration(
                        value_min=20, value_max=80, unit=TimeUnit.MINUTE
                    ),
                    frequency=1,
                    interval=1 * TimeUnit.DAY,
                ),
                {
                    interval("2023-03-04 00:00:00+01:00", "2023-03-04 23:59:59+01:00")
                },  # because we are looking for 20-80 minutes per day, 1 times a day
            ),
            (
                Timing(
                    duration=ValueDuration(
                        value_min=20, value_max=80, unit=TimeUnit.MINUTE
                    ),
                    frequency=2,
                    interval=1 * TimeUnit.DAY,
                ),
                set(),  # because we are looking for 20-80 minutes per day, 2 times a day
            ),
            (
                Timing(
                    duration=ValueDuration(
                        value_min=20, value_max=90, unit=TimeUnit.MINUTE
                    ),
                    frequency=2,
                    interval=1 * TimeUnit.DAY,
                ),
                {
                    interval("2023-03-04 00:00:00+01:00", "2023-03-04 23:59:59+01:00")
                },  # because we are looking for 20-80 minutes per day, 2 times a day
            ),
        ],
    )
    def test_timing_duration_interval(
        self,
        person_visit,
        db_session,
        concept,
        criterion_execute_func,
        observation_window,
        base_table,
        time_ranges,
        timing,
        expected_intervals,
    ):
        _, vo = person_visit[0]

        def criterion_execute_func_timing(
            concept: Concept,
            exclude: bool,
            value: ValueNumber | None = None,
        ):
            criterion = ProcedureOccurrence(
                exclude=False,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
                timing=timing,
                static=None,
            )
            if exclude:
                criterion = LogicalCriterionCombination.Not(
                    criterion, criterion.category
                )
            self.insert_criterion(db_session, criterion, observation_window)

            df = self.fetch_interval_result(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )
            df = df.query('interval_type=="POSITIVE"')

            return df_to_datetime_interval(
                df[["interval_start", "interval_end", "interval_type"]]
            )

        time_ranges = [TimeRange.from_tuple(tr[:2]) for tr in time_ranges]
        self.insert_occurrences(concept, db_session, vo, time_ranges)

        # run criterion against db
        intervals = criterion_execute_func_timing(concept=concept, exclude=False)

        assert set(intervals) == expected_intervals

    @pytest.mark.parametrize(
        "time_ranges",
        [
            [
                ("2023-02-28 12:00:00+01:00", "2023-03-02 12:00:00+01:00"),  # 48h
                (
                    "2023-03-25 12:00:00+01:00",
                    "2023-03-27 12:00:00+02:00",
                ),  # 47h - during DST change
                ("2023-03-30 12:00:00+02:00", "2023-04-01 12:00:00+02:00"),  # 48h
            ]
        ],
    )
    @pytest.mark.parametrize(
        "timing,expected_intervals",
        [
            (
                Timing(
                    duration=ValueDuration(value_min=16, unit=TimeUnit.HOUR),
                    frequency=1,
                    interval=1 * TimeUnit.DAY,
                ),
                {
                    # unmerged intervals
                    # interval("2023-02-28 00:00:00+01:00", "2023-02-28 23:59:59+01:00"),
                    # interval("2023-03-01 00:00:00+01:00", "2023-03-01 23:59:59+01:00"),
                    # interval("2023-03-02 00:00:00+01:00", "2023-03-02 23:59:59+01:00"),
                    # interval("2023-03-25 00:00:00+01:00", "2023-03-25 23:59:59+01:00"),
                    # interval(
                    #     "2023-03-26 00:00:00+01:00", "2023-03-26 23:59:59+02:00"
                    # ),  # DST change
                    # interval("2023-03-27 00:00:00+02:00", "2023-03-27 23:59:59+02:00"),
                    # interval("2023-03-30 00:00:00+02:00", "2023-03-30 23:59:59+02:00"),
                    # interval("2023-03-31 00:00:00+02:00", "2023-03-31 23:59:59+02:00"),
                    # interval("2023-04-01 00:00:00+02:00", "2023-04-01 23:59:59+02:00"),
                    # merged intervals
                    interval("2023-02-28 00:00:00+01:00", "2023-03-02 23:59:59+01:00"),
                    interval("2023-03-25 00:00:00+01:00", "2023-03-27 23:59:59+02:00"),
                    interval("2023-03-30 00:00:00+02:00", "2023-04-01 23:59:59+02:00"),
                },
            ),
        ],
    )
    def test_daylight_saving_time_handling(
        self,
        person_visit,
        db_session,
        concept,
        criterion_execute_func,
        observation_window,
        base_table,
        time_ranges,
        timing,
        expected_intervals,
    ):
        _, vo = person_visit[0]

        def criterion_execute_func_timing(
            concept: Concept,
            exclude: bool,
            value: ValueNumber | None = None,
        ):
            criterion = ProcedureOccurrence(
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
                timing=timing,
                static=None,
            )
            self.insert_criterion(db_session, criterion, observation_window)

            df = self.fetch_interval_result(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )
            df = df.query('interval_type=="POSITIVE"')

            return df_to_datetime_interval(
                df[["interval_start", "interval_end", "interval_type"]]
            )

        time_ranges = [TimeRange.from_tuple(tr[:2]) for tr in time_ranges]
        self.insert_occurrences(concept, db_session, vo, time_ranges)

        # run criterion against db
        intervals = criterion_execute_func_timing(concept=concept, exclude=False)

        assert set(intervals) == expected_intervals

    def test_serialization(self, concept):
        original = ProcedureOccurrence(
            exclude=False,
            category=CohortCategory.POPULATION,
            concept=concept,
            value=None,
            timing=Timing(
                duration=ValueDuration(value_min=16, unit=TimeUnit.HOUR),
                frequency=1,
                interval=1 * TimeUnit.DAY,
            ),
        )
        json = original.json()
        deserialized = ProcedureOccurrence.from_json(json)
        assert original == deserialized
