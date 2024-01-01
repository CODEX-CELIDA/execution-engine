import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.util import IntervalType as T


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
def test_trigger_interval_overlap_check(db_session, intervals, error_expected):
    """
    Tests that the trigger interval overlap check works as expected.
    """

    db_session.execute(
        text("SET session_replication_role = 'replica';")
    )  # Disable foreign key checks

    def insert_intervals():
        for interval in intervals:
            db_session.execute(
                ResultInterval.__table__.insert().values(
                    person_id=1,
                    run_id=1,
                    pi_pair_id=1,
                    criterion_id=1,
                    cohort_category=CohortCategory.POPULATION,
                    interval_start=interval["start"],
                    interval_end=interval["end"],
                    interval_type=interval["type"],
                )
            )

    if error_expected:
        with pytest.raises(DBAPIError):
            insert_intervals()
    else:
        insert_intervals()
