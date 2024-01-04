import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.celida.tables import (
    Criterion,
    ExecutionRun,
    PopulationInterventionPair,
    Recommendation,
    ResultInterval,
)
from execution_engine.omop.db.omop.tables import Person
from execution_engine.util.interval import IntervalType as T


@pytest.fixture
def setup_database(db_session):
    recommendation_id = 1
    run_id = 1
    pi_pair_id = 1
    criterion_id = 1
    person_id = 1

    start, end = datetime.datetime.now(), datetime.datetime.now()

    person = Person(
        person_id=person_id,
        gender_concept_id=0,
        year_of_birth=0,
        race_concept_id=0,
        ethnicity_concept_id=0,
    )
    db_session.add(person)
    db_session.commit()

    recommendation = Recommendation(
        recommendation_id=recommendation_id,
        recommendation_name="my_recommendation",
        recommendation_title="my_title",
        recommendation_url="https://example.com",
        recommendation_version="1.0",
        recommendation_hash=hash("my_recommendation"),
        recommendation_json="{}".encode(),
        create_datetime=datetime.datetime.now(),
    )
    db_session.add(recommendation)
    db_session.commit()

    run = ExecutionRun(
        run_id=run_id,
        observation_start_datetime=start,
        observation_end_datetime=end,
        run_datetime=datetime.datetime.now(),
        recommendation_id=recommendation_id,
    )
    db_session.add(run)
    db_session.commit()

    pi_pair = PopulationInterventionPair(
        pi_pair_id=pi_pair_id,
        recommendation_id=recommendation_id,
        pi_pair_url="https://example.com",
        pi_pair_name="my_pair",
        pi_pair_hash=hash("my_pai"),
    )
    db_session.add(pi_pair)
    db_session.commit()

    criterion = Criterion(
        criterion_id=criterion_id,
        criterion_name="my_criterion",
        criterion_description="my_description",
        criterion_hash=hash("my_criterion"),
    )
    db_session.add(criterion)
    db_session.commit()

    yield {
        "run_id": run_id,
        "pi_pair_id": pi_pair_id,
        "criterion_id": criterion_id,
        "person_id": person_id,
    }

    db_session.execute(text('TRUNCATE TABLE "cds_cdm"."person" CASCADE;'))
    db_session.commit()
    db_session.execute(text('TRUNCATE TABLE "celida"."execution_run" CASCADE;'))
    db_session.commit()
    db_session.execute(text('TRUNCATE TABLE "celida"."criterion" CASCADE;'))
    db_session.commit()
    db_session.execute(text('TRUNCATE TABLE "celida"."recommendation" CASCADE;'))
    db_session.commit()


@pytest.mark.parametrize(
    "intervals, error_expected",
    [
        (
            [
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
                {
                    "start": "2020-01-02 00:00:01+01:00",
                    "end": "2020-01-03 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
            ],
            False,
        ),
        (
            [
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
            ],
            True,
        ),  # same interval
        (
            [
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:01+01:00",
                    "type": T.POSITIVE,
                },
            ],
            True,
        ),  # overlap
        (
            [
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.POSITIVE,
                },
                {
                    "start": "2020-01-01 00:00:00+01:00",
                    "end": "2020-01-02 00:00:00+01:00",
                    "type": T.NEGATIVE,
                },
            ],
            True,
        ),  # same interval, different type
    ],
)
def test_trigger_interval_overlap_check(
    db_session, setup_database, intervals, error_expected
):
    """
    Tests that the trigger interval overlap check works as expected.
    """

    db_session.execute(
        text("SET session_replication_role = 'origin';")
    )  # Make sure triggers are enabled.

    def insert_intervals():
        for interval in intervals:
            db_session.execute(
                ResultInterval.__table__.insert().values(
                    person_id=setup_database["person_id"],
                    run_id=setup_database["run_id"],
                    pi_pair_id=setup_database["pi_pair_id"],
                    criterion_id=setup_database["criterion_id"],
                    cohort_category=CohortCategory.POPULATION,
                    interval_start=interval["start"],
                    interval_end=interval["end"],
                    interval_type=interval["type"],
                )
            )

    if error_expected:
        with pytest.raises(DBAPIError):
            insert_intervals()
        db_session.rollback()
    else:
        insert_intervals()
