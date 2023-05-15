import datetime

import pandas as pd
import pendulum
import pytest
from sqlalchemy import Column, Date, Integer, MetaData, Table, func

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.cdm import Person
from execution_engine.util import TimeRange
from tests.functions import create_visit


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


class TestCriterion:
    @pytest.fixture
    def visit_datetime(self) -> TimeRange:
        return TimeRange(
            name="visit", start="2023-03-01 09:36:24", end="2023-03-31 14:21:11"
        )

    @pytest.fixture
    def observation_window(self, visit_datetime: TimeRange) -> TimeRange:
        dt = visit_datetime.copy()
        dt.name = "observation"
        return dt

    @pytest.fixture
    def person(self, db_session):
        person = Person(
            person_id=1,
            gender_concept_id=0,
            year_of_birth=1990,
            month_of_birth=1,
            day_of_birth=1,
            race_concept_id=0,
            ethnicity_concept_id=0,
        )

        db_session.add(person)
        db_session.commit()

        return person

    @pytest.fixture
    def person_visit(self, person, visit_datetime, db_session):

        vo = create_visit(person.person_id, visit_datetime.start, visit_datetime.end)

        db_session.add(vo)
        db_session.commit()

        return [person, vo]

    @staticmethod
    def create_base_table(base_criterion, db_session, observation_window):
        base_table = to_table("base_table")
        query = base_criterion.sql_generate(base_table=base_table)
        query = base_criterion.sql_insert_into_table(query, base_table, temporary=True)
        db_session.execute(
            query,
            params=observation_window.dict(),
        )
        db_session.commit()

        return base_table

    @pytest.fixture
    def base_table(
        self,
        person_visit,
        db_session,
        base_criterion,
        observation_window,
    ):
        base_table = self.create_base_table(
            base_criterion, db_session, observation_window
        )

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
    def concept(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def criterion_class(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def occurrence_criterion(
        self,
        criterion_class,
        concept,
        base_table,
        db_session,
        observation_window,
    ):
        def _create_occurrence(exclude: bool):

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
                params=observation_window.dict(),
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        return _create_occurrence

    def invert_date_range(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        subtract: list[tuple[datetime.datetime, datetime.datetime]],
    ) -> set[datetime.date]:
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
    def date_points(
        times: list[datetime.datetime | datetime.date],
    ) -> set[datetime.date]:
        """
        Convert a list of datetimes to the corresponding set of (unique) dates.
        """
        return set([t.date() if isinstance(t, datetime.datetime) else t for t in times])

    def invert_date_points(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        subtract: list[datetime.date],
    ) -> set[datetime.date]:
        """
        Subtract a list of date points (set of days) from a date range.
        """
        main_dates_set = set(
            pendulum.period(start=start_datetime.date(), end=end_datetime.date()).range(
                "days"
            )
        )

        main_dates_set -= self.date_points(times=subtract)

        return main_dates_set

    @staticmethod
    def date_range(
        start_datetime: datetime.datetime, end_datetime: datetime.datetime
    ) -> set[datetime.date]:
        """
        Convert a start and end datetime to a set of all days inbetween.
        """
        return set(
            pendulum.period(start=start_datetime.date(), end=end_datetime.date()).range(
                "days"
            )
        )

    def date_ranges(
        self, time_ranges: list[tuple[datetime.datetime, datetime.datetime]]
    ) -> set[datetime.date]:
        """
        Convert a list of start/end datetimes to a set of all days inbetween each of the given datetime ranges
        """
        return set().union(
            *[
                self.date_range(start_datetime=tr[0], end_datetime=tr[1])
                for tr in time_ranges
            ]
        )
