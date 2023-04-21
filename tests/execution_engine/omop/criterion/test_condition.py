import datetime

import pandas as pd
import pendulum
import pytest
from sqlalchemy import Column, Date, Integer, MetaData, Table, func

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.cdm import Person
from tests.functions import create_condition, create_visit


class TestCondition:
    @pytest.fixture
    def visit_start_datetime(self) -> datetime.datetime:
        return pendulum.parse("2023-03-01 09:36:24")

    @pytest.fixture
    def visit_end_datetime(self) -> datetime.datetime:
        return pendulum.parse("2023-03-31 14:21:11")

    @pytest.fixture
    def person_visit(self, visit_start_datetime, visit_end_datetime, db_session):
        p = Person(
            person_id=1,
            gender_concept_id=0,
            year_of_birth=1990,
            month_of_birth=1,
            day_of_birth=1,
            race_concept_id=0,
            ethnicity_concept_id=0,
        )
        vo = create_visit(p, visit_start_datetime, visit_end_datetime)

        person_entries = [p, vo]

        db_session.add_all(person_entries)
        db_session.commit()

        return person_entries

    @pytest.fixture
    def base_table(
        self,
        person_visit,
        db_session,
        base_criterion,
        visit_start_datetime,
        visit_end_datetime,
    ):
        def to_table(name: str) -> Table:
            """
            Convert a name to a valid SQL table name.
            """
            metadata = MetaData()
            return Table(
                name,
                metadata,
                Column("person_id", Integer, primary_key=True),
                Column("valid_date", Date),
            )

        base_table = to_table("base_table")
        query = base_criterion.sql_generate(base_table=base_table)
        query = base_criterion.sql_insert_into_table(query, base_table, temporary=True)
        db_session.execute(
            query,
            params={
                "observation_start_datetime": visit_start_datetime,
                "observation_end_datetime": visit_end_datetime,
            },
        )
        db_session.commit()

        count = db_session.query(func.count(base_table.c.person_id)).scalar()
        assert (
            count > 0
        ), "Base table (active patients in period) should have at least one row."

        yield base_table

        base_table.drop(db_session.connection())

    @pytest.fixture
    def base_criterion(self):
        return PatientsActiveDuringPeriod("TestActivePatients")

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

    def invert_date_range(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        subtract: list[tuple[datetime.datetime, datetime.datetime]],
    ) -> set[datetime.datetime]:
        """
        Subtract a list of date ranges from a date range.
        """
        main_dates_set = self.date_range(
            start_datetime=start_datetime, end_datetime=end_datetime
        )

        for start, end in subtract:
            remove_dates_set = set(
                pendulum.period(start=start.date(), end=end.date()).range("days")
            )
            main_dates_set -= remove_dates_set

        return main_dates_set

    @staticmethod
    def date_range(
        start_datetime: datetime.datetime, end_datetime: datetime.datetime
    ) -> set[datetime.datetime]:
        return set(
            pendulum.period(start=start_datetime.date(), end=end_datetime.date()).range(
                "days"
            )
        )

    def date_ranges(
        self, time_ranges: list[tuple[datetime.datetime, datetime.datetime]]
    ) -> set[datetime.datetime]:
        return set().union(
            *[
                self.date_range(start_datetime=tr[0], end_datetime=tr[1])
                for tr in time_ranges
            ]
        )

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
