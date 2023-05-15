from abc import ABC

import pytest

from execution_engine.util import TimeRange
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set


class Occurrence(TestCriterion, ABC):
    def test_single_occurrence_single_time(
        self,
        person_visit,
        db_session,
        concept,
        occurrence_criterion,
        observation_window,
    ):
        _, vo = person_visit[0]

        time_ranges = [("2023-03-04 18:00:00", "2023-03-04 18:00:00")]

        self.perform_test(
            person_visit,
            concept,
            db_session,
            occurrence_criterion,
            observation_window,
            time_ranges,
        )

    def test_single_occurrence_single_day(
        self,
        person_visit,
        db_session,
        concept,
        occurrence_criterion,
        observation_window,
    ):
        _, vo = person_visit[0]

        time_ranges = [("2023-03-04 18:00:00", "2023-03-04 19:30:00")]

        self.perform_test(
            person_visit,
            concept,
            db_session,
            occurrence_criterion,
            observation_window,
            time_ranges,
        )

    def test_single_occurrence_multiple_days(
        self,
        person_visit,
        db_session,
        concept,
        occurrence_criterion,
        observation_window,
    ):
        time_ranges = [("2023-03-04 18:00:00", "2023-03-06 18:00:00")]

        self.perform_test(
            person_visit,
            concept,
            db_session,
            occurrence_criterion,
            observation_window,
            time_ranges,
        )

    @pytest.mark.parametrize(
        "time_ranges",
        [
            [  # non-overlapping
                ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
                ("2023-03-04 03:00:00", "2023-03-04 05:00:00"),
                ("2023-03-04 06:00:00", "2023-03-04 08:00:00"),
            ],
            [  # exact overlap
                ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
                ("2023-03-04 02:00:00", "2023-03-04 04:00:00"),
                ("2023-03-04 04:00:00", "2023-03-04 06:00:00"),
            ],
            [  # overlap by some margin
                ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
                ("2023-03-04 01:30:00", "2023-03-04 03:30:00"),
                ("2023-03-04 04:00:00", "2023-03-04 06:00:00"),
            ],
            [  # full overlap
                ("2023-03-04 00:00:00", "2023-03-04 06:00:00"),
                ("2023-03-04 01:00:00", "2023-03-04 02:00:00"),
                ("2023-03-04 03:00:00", "2023-03-04 04:00:00"),
            ],
        ],
    )
    def test_multiple_occurrences_single_day(
        self,
        person_visit,
        db_session,
        concept,
        occurrence_criterion,
        observation_window,
        time_ranges,
    ):
        self.perform_test(
            person_visit,
            concept,
            db_session,
            occurrence_criterion,
            observation_window,
            time_ranges,
        )

    @pytest.mark.parametrize(
        "time_ranges",
        [
            [  # non-overlapping
                ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),
                ("2023-03-07 10:00:00", "2023-03-09 18:00:00"),
            ],
            [  # exact overlap
                ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                ("2023-03-03 16:00:00", "2023-03-05 23:59:00"),
                ("2023-03-05 23:59:00", "2023-03-08 11:00:00"),
            ],
            [  # overlap by some margin
                ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),
                ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),
            ],
            [  # full overlap
                ("2023-03-01 08:00:00", "2023-03-09 18:00:00"),
                ("2023-03-03 10:00:00", "2023-03-05 12:00:00"),
                ("2023-03-06 14:00:00", "2023-03-08 16:00:00"),
            ],
        ],
    )
    def test_multiple_occurrences_multiple_days(
        self,
        person_visit,
        db_session,
        concept,
        occurrence_criterion,
        observation_window,
        time_ranges,
    ):
        self.perform_test(
            person_visit,
            concept,
            db_session,
            occurrence_criterion,
            observation_window,
            time_ranges,
        )

    @pytest.mark.parametrize(
        "test_cases",
        [
            (
                [
                    {
                        "time_range": [  # non-overlapping
                            ("2023-03-03 08:00:00", "2023-03-03 16:00:00"),
                            ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),
                            ("2023-03-08 10:00:00", "2023-03-09 18:00:00"),
                        ],
                        "expected": {
                            "2023-03-03",
                            "2023-03-04",
                            "2023-03-05",
                            "2023-03-06",
                            "2023-03-08",
                            "2023-03-09",
                        },
                    },
                    {
                        "time_range": [  # exact overlap
                            ("2023-03-01 08:00:00", "2023-03-02 16:00:00"),
                            ("2023-03-02 16:00:00", "2023-03-03 23:59:00"),
                            ("2023-03-03 23:59:00", "2023-03-04 11:00:00"),
                        ],
                        "expected": {
                            "2023-03-01",
                            "2023-03-02",
                            "2023-03-03",
                            "2023-03-04",
                        },
                    },
                    {
                        "time_range": [  # overlap by some margin
                            ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                            ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),
                            ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),
                        ],
                        "expected": {
                            "2023-03-01",
                            "2023-03-02",
                            "2023-03-03",
                            "2023-03-04",
                            "2023-03-05",
                            "2023-03-06",
                            "2023-03-07",
                            "2023-03-08",
                        },
                    },
                ]
            )
        ],
    )
    def test_multiple_persons(
        self, person_visit, db_session, concept, occurrence_criterion, test_cases
    ):
        vos = [pv[1] for pv in person_visit]

        assert len(vos) == len(
            test_cases
        ), "Number of test cases must match number of persons"

        for test_case, vo in zip(test_cases, vos):
            time_ranges = [TimeRange.from_tuple(tr) for tr in test_case["time_range"]]

            self.insert_occurrences(concept, db_session, vo, time_ranges)

        # run criterion against db
        df = occurrence_criterion(exclude=False)

        for test_case, vo in zip(test_cases, vos):
            assert set(
                df.query(f"person_id=={vo.person_id}")["valid_date"].dt.date
            ) == date_set(test_case["expected"])

    def insert_occurrences(self, concept, db_session, visit_occurrence, time_ranges):

        for tr in time_ranges:
            c = self.create_occurrence(
                visit_occurrence=visit_occurrence,
                concept_id=concept.concept_id,
                start_datetime=tr.start,
                end_datetime=tr.end,
            )
            db_session.add(c)

        db_session.commit()

    def perform_test(
        self,
        person_visit,
        concept,
        db_session,
        occurrence_criterion,
        observation_window,
        time_ranges,
    ):
        p, vo = person_visit[0]

        time_ranges = [TimeRange.from_tuple(tr) for tr in time_ranges]

        self.insert_occurrences(concept, db_session, vo, time_ranges)

        # run criterion against db
        df = occurrence_criterion(exclude=False)
        valid_daterange = self.date_ranges(time_ranges)
        assert (
            set(df.query(f"person_id=={p.person_id}")["valid_date"].dt.date)
            == valid_daterange
        )

        df = occurrence_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            time_range=observation_window,
            subtract=time_ranges,
        )
        assert (
            set(df.query(f"person_id=={p.person_id}")["valid_date"].dt.date)
            == valid_daterange
        )
