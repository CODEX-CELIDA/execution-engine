import logging

import pytest
from sqlalchemy import insert, select, text

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.omop.db.cdm import Person

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def create_test_data():
    """Let's create the test data with the three witches names."""
    test_stmts = []

    test_stmts.append(
        insert(Person).values(
            gender_concept_id=0,
            year_of_birth=1980,
            month_of_birth=3,
            day_of_birth=3,
            race_concept_id=0,
            ethnicity_concept_id=0,
        )
    )

    return test_stmts


def test_persons(db_session, create_test_data):
    s = db_session()

    s.execute(
        text("SET session_replication_role = 'replica';")
    )  # Disable foreign key checks

    for obj in create_test_data:
        s.execute(obj)
    s.commit()
    logger.info("Added test data to the database.")

    s.execute(select(Person)).all()

    s.close()

    assert True
