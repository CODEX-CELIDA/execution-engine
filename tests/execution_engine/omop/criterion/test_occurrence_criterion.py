from abc import ABC, abstractmethod

import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion


class Occurrence(TestCriterion, ABC):
    @pytest.fixture
    def concept(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def criterion_class(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @abstractmethod
    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def occurrence_criterion(
        self, criterion_class, concept, base_table, db_session, person_visit
    ):
        def _create_occurrence(exclude: bool):
            p, vo = person_visit

            criterion = criterion_class(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=None,
                static=None,
            )

            query = criterion.sql_generate(base_table=base_table)

            df = pd.read_sql(
                query,
                db_session.connection(),
                params={
                    "observation_start_datetime": vo.visit_start_datetime,
                    "observation_end_datetime": vo.visit_end_datetime,
                },
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        return _create_occurrence

    def test_single_occurrence_single_time(
        self, person_visit, db_session, concept, occurrence_criterion
    ):
        _, vo = person_visit

        time_ranges = [("2023-03-04 18:00:00", "2023-03-04 18:00:00")]

        self.perform_test(
            person_visit, concept, db_session, occurrence_criterion, time_ranges
        )

    def test_single_occurrence_single_day(
        self, person_visit, db_session, concept, occurrence_criterion
    ):
        _, vo = person_visit

        time_ranges = [("2023-03-04 18:00:00", "2023-03-04 19:30:00")]

        self.perform_test(
            person_visit, concept, db_session, occurrence_criterion, time_ranges
        )

    def test_single_occurrence_multiple_days(
        self, person_visit, db_session, concept, occurrence_criterion
    ):

        time_ranges = [("2023-03-04 18:00:00", "2023-03-06 18:00:00")]

        self.perform_test(
            person_visit, concept, db_session, occurrence_criterion, time_ranges
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
        time_ranges,
    ):
        self.perform_test(
            person_visit, concept, db_session, occurrence_criterion, time_ranges
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
        time_ranges,
    ):
        self.perform_test(
            person_visit, concept, db_session, occurrence_criterion, time_ranges
        )

    def perform_test(
        self, person_visit, concept, db_session, occurrence_criterion, time_ranges
    ):
        _, vo = person_visit

        time_ranges = [
            (pendulum.parse(start), pendulum.parse(end)) for start, end in time_ranges
        ]

        for start, end in time_ranges:
            c = self.create_occurrence(
                visit_occurrence=vo,
                concept_id=concept.concept_id,
                start_datetime=start,
                end_datetime=end,
            )
            db_session.add(c)

        db_session.commit()

        # run criterion against db
        df = occurrence_criterion(exclude=False)
        valid_daterange = self.date_ranges(time_ranges)
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = occurrence_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=time_ranges,
        )
        assert set(df["valid_date"].dt.date) == valid_daterange
