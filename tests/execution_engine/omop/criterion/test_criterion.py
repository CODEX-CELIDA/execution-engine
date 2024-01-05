import datetime
from contextlib import contextmanager
from typing import Iterable, Sequence

import pandas as pd
import pendulum
import pytest
from sqlalchemy import Column, Date, Integer, MetaData, Table, select

import execution_engine.omop.db.celida.tables as celida_tables
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.celida.views import (
    full_day_coverage,
    interval_result,
    partial_day_coverage,
)
from execution_engine.omop.db.omop.tables import Person
from execution_engine.task import (  # noqa: F401     -- required for the mock.patch below
    process,
    runner,
    task,
)
from execution_engine.util import cohort_logic
from execution_engine.util.db import add_result_insert
from execution_engine.util.types import TimeRange
from execution_engine.util.value import ValueConcept, ValueNumber
from tests._fixtures.omop_fixture import celida_recommendation
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
    result_table = celida_tables.ResultInterval.__table__
    run_id = 1234
    criterion_id = 5678
    pi_pair_id = 9012
    recommendation_id = 3456
    merge_intervals_before_writing: bool = False

    @pytest.fixture(autouse=True)
    def create_recommendation_run(self, db_session, observation_window) -> None:
        with celida_recommendation(
            db_session,
            observation_window,
            recommendation_id=self.recommendation_id,
            run_id=self.run_id,
            pi_pair_id=self.pi_pair_id,
            criterion_id=self.criterion_id,
        ):
            yield

    @pytest.fixture
    def visit_datetime(self) -> TimeRange:
        return TimeRange(
            name="visit", start="2023-03-01 09:36:24Z", end="2023-03-31 14:21:11Z"
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

    @classmethod
    @contextmanager
    def execute_base_criterion(cls, base_criterion, db_session, observation_window):
        query = base_criterion.create_query()
        cls.register_criterion(base_criterion, db_session)

        # add base table patients to results table, so they can be used when combining statements (execution_map)
        query = add_result_insert(
            query,
            pi_pair_id=cls.pi_pair_id,
            criterion_id=base_criterion.id,
            cohort_category=CohortCategory.BASE,
        )

        db_session.execute(
            query,
            params={"run_id": cls.run_id} | observation_window.dict(),
        )

        db_session.commit()

        try:
            yield
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.query(celida_tables.ResultInterval).delete()
            db_session.commit()

    @pytest.fixture
    def base_table(
        self,
        person_visit,
        db_session,
        base_criterion,
        observation_window,
    ):
        with self.execute_base_criterion(
            base_criterion, db_session, observation_window
        ):
            yield

    @classmethod
    def register_criterion(cls, criterion: Criterion, db_session):
        """
        Register a criterion in the database.
        """

        exists = db_session.query(
            db_session.query(celida_tables.Criterion)
            .filter_by(criterion_id=criterion.id)
            .exists()
        ).scalar()

        if not exists:
            new_criterion = celida_tables.Criterion(
                criterion_id=criterion.id,
                criterion_name=criterion.description(),
                criterion_description=criterion.description(),
                criterion_hash=hash(criterion),
            )
            db_session.add(new_criterion)
            db_session.commit()

    @classmethod
    def register_population_intervention_pair(cls, id_, name, db_session):
        """
        Register a criterion in the database.
        """

        exists = db_session.query(
            db_session.query(celida_tables.PopulationInterventionPair)
            .filter_by(pi_pair_id=id_)
            .exists()
        ).scalar()

        if not exists:
            new_criterion = celida_tables.PopulationInterventionPair(
                pi_pair_id=id_,
                recommendation_id=-1,
                pi_pair_url="https://example.com",
                pi_pair_name=name,
                pi_pair_hash=hash(name),
            )
            db_session.add(new_criterion)
            db_session.commit()

    @pytest.fixture
    def base_criterion(self):
        c = PatientsActiveDuringPeriod("TestActivePatients")
        c.id = -1
        return c

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

    def insert_criterion(self, db_session, criterion, observation_window: TimeRange):
        criterion.id = self.criterion_id
        self.register_criterion(criterion, db_session)

        query = criterion.create_query()

        result = pd.read_sql(
            query, con=db_session.connection(), params=observation_window.dict()
        )
        result = process.merge_intervals_negative_dominant(result, by=["person_id"])

        result = result.assign(
            criterion_id=self.criterion_id,
            pi_pair_id=self.pi_pair_id,
            run_id=self.run_id,
            cohort_category=CohortCategory.POPULATION,
        )

        db_session.execute(
            celida_tables.ResultInterval.__table__.insert(),
            result.to_dict(orient="records"),
        )

        db_session.commit()

    def insert_criterion_combination(
        self, db_session, combination, base_criterion, observation_window: TimeRange
    ):
        graph = ExecutionGraph.from_criterion_combination(combination, base_criterion)

        # we need to add a NoDataPreservingAnd node to insert NEGATIVE intervals
        sink_node = graph.sink_node(CohortCategory.POPULATION)
        p_combination_node = cohort_logic.NoDataPreservingAnd(
            sink_node, category=CohortCategory.POPULATION
        )

        graph.add_node(
            p_combination_node, store_result=True, category=CohortCategory.POPULATION
        )
        graph.add_edge(sink_node, p_combination_node)

        graph.set_sink_nodes_store(bind_params={"pi_pair_id": self.pi_pair_id})

        params = observation_window.dict() | {"run_id": self.run_id}

        task_runner = runner.SequentialTaskRunner(graph)
        task_runner.run(params)

    def fetch_full_day_result(
        self,
        db_session,
        pi_pair_id: int | None,
        criterion_id: int | None,
        category: CohortCategory | None,
    ):
        df = self._fetch_result_view(
            full_day_coverage, db_session, pi_pair_id, criterion_id, category
        )
        df["valid_date"] = pd.to_datetime(df["valid_date"])

        return df

    def fetch_partial_day_result(
        self,
        db_session,
        pi_pair_id: int | None,
        criterion_id: int | None,
        category: CohortCategory | None,
    ):
        df = self._fetch_result_view(
            partial_day_coverage, db_session, pi_pair_id, criterion_id, category
        )
        df["valid_date"] = pd.to_datetime(df["valid_date"])

        return df

    def fetch_interval_result(
        self,
        db_session,
        pi_pair_id: int | None,
        criterion_id: int | None,
        category: CohortCategory | None,
    ):
        return self._fetch_result_view(
            interval_result, db_session, pi_pair_id, criterion_id, category
        )

    def _fetch_result_view(
        self,
        view,
        db_session,
        pi_pair_id: int | None,
        criterion_id: int | None,
        category: CohortCategory | None,
    ):
        stmt = select(view).where(view.c.run_id == self.run_id)

        if criterion_id is not None:
            stmt = stmt.where(view.c.criterion_id == criterion_id)
        else:
            stmt = stmt.where(view.c.criterion_id.is_(None))

        if pi_pair_id is not None:
            stmt = stmt.where(view.c.pi_pair_id == pi_pair_id)
        else:
            stmt = stmt.where(view.c.pi_pair_id.is_(None))

        if category is not None:
            stmt = stmt.where(view.c.cohort_category == category)

        df = pd.read_sql(
            stmt,
            db_session.connection(),
            params={"run_id": self.run_id},
        )

        return df

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

            self.insert_criterion(db_session, criterion, observation_window)

            df = self.fetch_full_day_result(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )

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
