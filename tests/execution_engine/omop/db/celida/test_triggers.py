import pendulum
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.omop.db.omop.tables import Person
from execution_engine.util.interval import IntervalType as T
from execution_engine.util.types import TimeRange
from tests._fixtures.omop_fixture import celida_recommendation


@pytest.fixture
def person(db_session):
    person_id = 1
    person = Person(
        person_id=person_id,
        gender_concept_id=0,
        year_of_birth=0,
        race_concept_id=0,
        ethnicity_concept_id=0,
    )

    db_session.add(person)
    db_session.commit()

    yield person

    db_session.execute(text('TRUNCATE TABLE "cds_cdm"."person" CASCADE;'))
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
def test_trigger_interval_overlap_check(db_session, intervals, error_expected, person):
    """
    Tests that the trigger interval overlap check works as expected.
    """

    def insert_intervals(rec):
        for interval in intervals:
            db_session.execute(
                ResultInterval.__table__.insert().values(
                    person_id=person.person_id,
                    run_id=rec["run_id"],
                    pi_pair_id=rec["pi_pair_id"],
                    criterion_id=rec["criterion_id"],
                    cohort_category=CohortCategory.POPULATION,
                    interval_start=interval["start"],
                    interval_end=interval["end"],
                    interval_type=interval["type"],
                )
            )

    observation_window = TimeRange(start=pendulum.now("UTC"), end=pendulum.now("UTC"))

    with celida_recommendation(db_session, observation_window) as rec:
        if error_expected:
            with pytest.raises(DBAPIError):
                insert_intervals(rec)
            db_session.rollback()
        else:
            insert_intervals(rec)
