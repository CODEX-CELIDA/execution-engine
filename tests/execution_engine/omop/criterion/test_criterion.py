import datetime
from contextlib import contextmanager
from typing import Iterable, Sequence

import pandas as pd
import pendulum
import pytest
from sqlalchemy import Column, Date, Integer, MetaData, Table, func, text

from execution_engine.constants import CohortCategory
from execution_engine.omop.cohort import add_result_insert
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.cdm import Person
from execution_engine.omop.db.celida.tables import (
    RecommendationResultInterval,
    RecommendationRun,
)
from execution_engine.omop.db.celida.views import partial_day_coverage
from execution_engine.util import TimeRange, ValueConcept, ValueNumber
from tests._testdata import concepts
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


def date_set(tc: Iterable):
    """
    Convert an iterable of timestamps to a set of dates.
    """
    return set(pendulum.parse(t).date() for t in tc)


class TestCriterion:
    result_table = RecommendationResultInterval.__table__
    result_view = partial_day_coverage

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
    def person(self, db_session, visit_datetime: TimeRange, n: int = 3):
        assert (
            0 < n < visit_datetime.duration.days / 2
        )  # because each further person's visit is 2 days shorter

        persons = [
            Person(
                person_id=i + 1,
                gender_concept_id=[
                    concepts.GENDER_MALE,
                    concepts.GENDER_FEMALE,
                    concepts.UNKNOWN,
                ][i % 3],
                year_of_birth=1980 + i,
                month_of_birth=1,
                day_of_birth=1,
                race_concept_id=0,
                ethnicity_concept_id=0,
            )
            for i in range(n)
        ]

        db_session.add_all(persons)
        db_session.commit()

        return persons

    @pytest.fixture
    def person_visit(self, person, visit_datetime, db_session):
        vos = [
            create_visit(
                person_id=p.person_id,
                visit_start_datetime=visit_datetime.start + datetime.timedelta(days=i),
                visit_end_datetime=visit_datetime.end - datetime.timedelta(days=i),
                visit_concept_id=concepts.INTENSIVE_CARE,
            )
            for i, p in enumerate(person)
        ]

        db_session.add_all(vos)
        db_session.commit()

        return list(zip(person, vos))

    @staticmethod
    @contextmanager
    def execute_base_criterion(base_criterion, db_session, observation_window):
        db_session.execute(
            text("SET session_replication_role = 'replica';")
        )  # Disable foreign key checks

        RECOMMENDATION_RUN_ID = 1234

        t = RecommendationRun.__table__
        db_session.execute(
            t.insert(),
            [
                {
                    "recommendation_run_id": RECOMMENDATION_RUN_ID,
                    "observation_start_datetime": observation_window.start,
                    "observation_end_datetime": observation_window.end,
                    "run_datetime": datetime.datetime.now(),
                    "cohort_definition_id": 1,
                }
            ],
        )
        query = base_criterion.create_query()

        # add base table patients to results table, so they can be used when combining statements (execution_map)
        query = add_result_insert(
            query,
            plan_id=None,
            criterion_id=None,
            cohort_category=CohortCategory.BASE,
        )

        db_session.execute(
            query, params={"run_id": RECOMMENDATION_RUN_ID} | observation_window.dict()
        )

        db_session.commit()

        try:
            yield
        finally:
            db_session.execute(text("SET session_replication_role = 'origin';"))
            db_session.query(RecommendationResultInterval).delete()
            db_session.query(RecommendationRun).delete()

    @pytest.fixture
    def base_table(
        self,
        person_visit,
        db_session,
        base_criterion,
        observation_window,
    ):
        base_table = self.execute_base_criterion(
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
    def criterion_execute_func(
        self, base_table, db_session, criterion_class, observation_window
    ):
        def _create_value(
            concept: Concept,
            exclude: bool,
            value: ValueNumber | ValueConcept | None = None,
        ) -> pd.DataFrame:
            criterion = criterion_class(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
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

        return _create_value

    def invert_date_range(
        self,
        time_range: TimeRange,
        subtract: list[TimeRange],
    ) -> set[datetime.date]:
        """
        Subtract a list of date ranges from a date range.
        """
        main_dates_set = time_range.date_range()

        for tr in subtract:
            main_dates_set -= tr.date_range()

        return main_dates_set

    @staticmethod
    def date_points(
        times: Sequence[datetime.datetime | datetime.date | str],
    ) -> set[datetime.date]:
        """
        Convert a list of datetimes to the corresponding set of (unique) dates.
        """
        res = []
        for t in times:
            if isinstance(t, datetime.datetime):
                res.append(t.date())
            elif isinstance(t, datetime.date):
                res.append(t)
            else:
                res.append(pendulum.parse(t).date())

        return set(res)

    def invert_date_points(
        self,
        time_range: TimeRange,
        subtract: list[datetime.date],
    ) -> set[datetime.date]:
        """
        Subtract a list of date points (set of days) from a date range.
        """
        main_dates_set = time_range.date_range()
        main_dates_set -= self.date_points(times=subtract)

        return main_dates_set

    def date_ranges(self, time_ranges: list[TimeRange]) -> set[datetime.date]:
        """
        Convert a list of start/end datetimes to a set of all days inbetween each of the given datetime ranges
        """
        return set().union(*[tr.date_range() for tr in time_ranges])
