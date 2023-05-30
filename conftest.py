import os
from glob import glob
from typing import Any

import pytest
from pytest_postgresql.janitor import DatabaseJanitor

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "")
    for fixture_file in glob("tests/_fixtures/[!__]*.py", recursive=True)
]


def postgres_janitor() -> DatabaseJanitor:
    """
    Create a janitor for postgresql.

    The janitor is used to create and destroy the database that is used in testing.

    :return: DatabaseJanitor
    """
    pg_user = os.environ["OMOP__USER"]
    pg_pass = os.environ["OMOP__PASSWORD"]
    pg_host = os.environ["OMOP__HOST"]
    pg_port = os.environ["OMOP__PORT"]
    pg_name = os.environ["OMOP__DATABASE"]

    janitor = DatabaseJanitor(
        user=pg_user,
        host=pg_host,
        port=pg_port,
        dbname=pg_name,
        password=pg_pass,
        version="16",
    )

    return janitor


def init_postgres(config):  # type: ignore
    """
    Initialize the postgres database.

    This function is called by pytest before the tests are run.
    - Set the environment variables that are used by the OMOPSQLClient
    - Drops the test database if it exists
    - Create the test database
    """

    def getvalue(name):  # type: ignore
        return config.getoption(name) or config.getini(name)

    os.environ["OMOP__USER"] = getvalue("postgresql_user")
    os.environ["OMOP__PASSWORD"] = getvalue("postgresql_password")
    os.environ["OMOP__HOST"] = getvalue("postgresql_host")
    os.environ["OMOP__PORT"] = str(getvalue("postgresql_port"))
    os.environ["OMOP__DATABASE"] = getvalue("postgresql_dbname")
    os.environ["OMOP__SCHEMA"] = "cds_cdm"

    janitor = postgres_janitor()

    # Drop the database if it already exists (e.g. from a previous interrupted test run)
    with janitor.cursor() as cur:
        db_exists = cur.execute(
            """SELECT EXISTS(
            SELECT datname FROM pg_catalog.pg_database WHERE datname = %(dbname)s
        )""",
            params={"dbname": getvalue("postgresql_dbname")},
        ).fetchone()[0]

    if db_exists:
        janitor.drop()

    janitor.init()


def pytest_sessionstart(session):  # type: ignore
    """
    Initialize the postgres database before the tests are run.
    """
    init_postgres(session.config)


def pytest_sessionfinish(session):  # type: ignore
    """
    Drop the postgres database after the tests are run.
    """
    janitor = postgres_janitor()
    janitor.drop()


def pytest_addoption(parser):  # type: ignore
    """
    Add command line options to pytest.

    See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    parser.addoption(
        "--run-slow-tests", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-recommendation-tests",
        action="store_true",
        default=False,
        help="run recommendation tests",
    )


def pytest_configure(config):  # type: ignore
    """
    Add custom markers to pytest.

    See https://docs.pytest.org/en/latest/example/markers.html#marking-test-functions-and-selecting-them-for-a-run
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "recommendation: mark test as a recommendation test"
    )


def pytest_collection_modifyitems(config, items):  # type: ignore
    """
    Skip slow tests if --run-slow-tests is not given.

    See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    if not config.getoption("--run-slow-tests"):
        # --run-slow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --run-slow-tests option to run")

        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-recommendation-tests"):
        # --run-recommendation-tests not given in cli: skip recommendation tests
        skip_recommendation = pytest.mark.skip(
            reason="need --run-recommendation-tests option to run"
        )

        for item in items:
            if "recommendation" in item.keywords:
                item.add_marker(skip_recommendation)


def pytest_assertrepr_compare(
    op: str,
    left: Any,
    right: Any,
) -> list[str] | None:
    """
    Custom error message for RecommendationCriteriaCombination.
    """
    from tests.recommendation.test_recommendation_base import (
        RecommendationCriteriaCombination,
    )

    if (
        isinstance(left, RecommendationCriteriaCombination)
        and isinstance(right, RecommendationCriteriaCombination)
        and op == "=="
    ):
        return left.comparison_report(right)

    return None


@pytest.fixture
def run_slow_tests(request) -> bool:  # type: ignore
    """
    Fixture to determine if slow tests should be run.
    """
    return request.config.getoption("--run-slow-tests")


@pytest.fixture
def run_recommendation_tests(request) -> bool:  # type: ignore
    """
    Fixture to determine if recommendation tests should be run.
    """
    return request.config.getoption("--run-recommendation-tests")
