import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.functions import create_condition


class TestCondition(TestCriterion):
    @pytest.fixture
    def concept_covid19(self):
        return Concept(
            concept_id=37311061,
            concept_name="COVID-19",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            concept_code="840539006",
            invalid_reason=None,
        )

    @pytest.fixture
    def condition_criterion(
        self, concept_covid19, base_table, db_session, person_visit
    ):
        def _create_criterion(exclude: bool):
            p, vo = person_visit

            criterion = ConditionOccurrence(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept_covid19,
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

        return _create_criterion

    def test_single_condition_single_time(
        self, person_visit, db_session, concept_covid19, condition_criterion
    ):
        _, vo = person_visit

        start_datetime = pendulum.parse("2023-03-04 18:00:00")
        end_datetime = pendulum.parse("2023-03-04 18:00:00")

        c = create_condition(
            vo=vo,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=start_datetime,
            condition_end_datetime=end_datetime,
        )

        db_session.add(c)
        db_session.commit()

        # run criterion against db
        df = condition_criterion(exclude=False)
        valid_daterange = self.date_range(
            start_datetime=start_datetime, end_datetime=end_datetime
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = condition_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=[(start_datetime, end_datetime)],
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

    def test_single_condition_single_day(
        self, person_visit, db_session, concept_covid19, condition_criterion
    ):
        _, vo = person_visit

        start_datetime = pendulum.parse("2023-03-04 18:00:00")
        end_datetime = pendulum.parse("2023-03-04 19:30:00")

        c = create_condition(
            vo=vo,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=start_datetime,
            condition_end_datetime=end_datetime,
        )

        db_session.add(c)
        db_session.commit()

        # run criterion against db
        df = condition_criterion(exclude=False)
        valid_daterange = self.date_range(
            start_datetime=start_datetime, end_datetime=end_datetime
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = condition_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=[(start_datetime, end_datetime)],
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

    def test_single_condition_multiple_days(
        self, person_visit, db_session, concept_covid19, condition_criterion
    ):
        _, vo = person_visit

        start_datetime = pendulum.parse("2023-03-04 18:00:00")
        end_datetime = pendulum.parse("2023-03-06 18:00:00")

        c = create_condition(
            vo=vo,
            condition_concept_id=concept_covid19.concept_id,
            condition_start_datetime=start_datetime,
            condition_end_datetime=end_datetime,
        )

        db_session.add(c)
        db_session.commit()

        # run criterion against db
        df = condition_criterion(exclude=False)
        valid_daterange = self.date_range(
            start_datetime=start_datetime, end_datetime=end_datetime
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = condition_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=[(start_datetime, end_datetime)],
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

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
    def test_multiple_conditions_single_day(
        self,
        person_visit,
        db_session,
        concept_covid19,
        condition_criterion,
        time_ranges,
    ):
        _, vo = person_visit

        time_ranges = [
            (pendulum.parse(start), pendulum.parse(end)) for start, end in time_ranges
        ]

        for start, end in time_ranges:
            c = create_condition(
                vo=vo,
                condition_concept_id=concept_covid19.concept_id,
                condition_start_datetime=start,
                condition_end_datetime=end,
            )

            db_session.add(c)

        db_session.commit()

        # run criterion against db
        df = condition_criterion(exclude=False)
        valid_daterange = self.date_ranges(time_ranges)
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = condition_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=time_ranges,
        )
        assert set(df["valid_date"].dt.date) == valid_daterange

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
    def test_multiple_conditions_multiple_days(
        self,
        person_visit,
        db_session,
        concept_covid19,
        condition_criterion,
        time_ranges,
    ):
        _, vo = person_visit

        time_ranges = [
            (pendulum.parse(start), pendulum.parse(end)) for start, end in time_ranges
        ]

        for start, end in time_ranges:
            c = create_condition(
                vo=vo,
                condition_concept_id=concept_covid19.concept_id,
                condition_start_datetime=start,
                condition_end_datetime=end,
            )

            db_session.add(c)

        db_session.commit()

        # run criterion against db
        df = condition_criterion(exclude=False)
        valid_daterange = self.date_ranges(time_ranges)
        assert set(df["valid_date"].dt.date) == valid_daterange

        df = condition_criterion(exclude=True)
        valid_daterange = self.invert_date_range(
            start_datetime=vo.visit_start_datetime,
            end_datetime=vo.visit_end_datetime,
            subtract=time_ranges,
        )
        assert set(df["valid_date"].dt.date) == valid_daterange
