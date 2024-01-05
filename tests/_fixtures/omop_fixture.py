import datetime
import logging
import os
from contextlib import contextmanager
from urllib.parse import quote

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm.session import sessionmaker

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.omop.db.celida.tables import (
    Criterion,
    ExecutionRun,
    PopulationInterventionPair,
    Recommendation,
)
from execution_engine.util.types import TimeRange

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIMEZONE = "Europe/Berlin"


@contextmanager
def disable_postgres_trigger(conn):
    conn.execute(text("SET session_replication_role = 'replica';"))
    conn.commit()

    yield

    conn.execute(text("SET session_replication_role = 'origin';"))
    conn.commit()


@pytest.fixture(scope="session")
def db_setup():
    """Database Session for SQLAlchemy."""

    pg_user = os.environ["OMOP__USER"]
    pg_password = os.environ["OMOP__PASSWORD"]
    pg_host = os.environ["OMOP__HOST"]
    pg_port = os.environ["OMOP__PORT"]
    pg_db = os.environ["OMOP__DATABASE"]

    connection_str = f"postgresql+psycopg://{quote(pg_user)}:{quote(pg_password)}@{pg_host}:{pg_port}/{pg_db}"
    engine = create_engine(connection_str)

    @event.listens_for(engine, "connect")
    def set_timezone(dbapi_connection, connection_record) -> None:
        """
        Set the timezone for the database connection.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute(
            "SELECT set_config('TIMEZONE', %(timezone)s, false)",
            {"timezone": TIMEZONE},
        )
        cursor.close()

    with engine.connect() as con:
        if not con.dialect.has_schema(con, "celida"):
            con.execute(sqlalchemy.schema.CreateSchema("celida"))
        if not con.dialect.has_schema(con, "cds_cdm"):
            con.execute(sqlalchemy.schema.CreateSchema("cds_cdm"))

        with disable_postgres_trigger(con):
            metadata.create_all(con)
            logger.info("Inserting test data into the database.")

            for table in [
                "concept",
                "concept_relationship",
                "concept_ancestor",
                "drug_strength",
            ]:
                df = pd.read_csv(
                    f"tests/_testdata/omop_cdm/{table}.csv.gz",
                    na_values=[""],
                    keep_default_na=False,
                )
                for c in df.columns:
                    if "_date" in c:
                        df[c] = pd.to_datetime(df[c])
                df.to_sql(table, con, schema="cds_cdm", if_exists="append", index=False)

            con.commit()

    yield sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db_session(db_setup):
    session = db_setup()
    try:
        yield session
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.execute(text('TRUNCATE TABLE "cds_cdm"."person" CASCADE;'))
        session.execute(text('TRUNCATE TABLE "celida"."recommendation" CASCADE;'))
        session.execute(text('TRUNCATE TABLE "celida"."execution_run" CASCADE;'))
        session.execute(
            text('TRUNCATE TABLE "celida"."population_intervention_pair" CASCADE;')
        )
        session.execute(text('TRUNCATE TABLE "celida"."criterion" CASCADE;'))
        session.commit()
        session.commit()


@contextmanager
def celida_recommendation(
    db_session,
    observation_window: TimeRange,
    recommendation_id=12,
    run_id=34,
    pi_pair_id=56,
    criterion_id=78,
):
    try:
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
            observation_start_datetime=observation_window.start,
            observation_end_datetime=observation_window.end,
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
            pi_pair_hash=hash("my_pair"),
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
            "recommendation_id": recommendation_id,
            "pi_pair_id": pi_pair_id,
            "criterion_id": criterion_id,
        }
    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.execute(text('TRUNCATE TABLE "celida"."recommendation" CASCADE;'))
        db_session.execute(text('TRUNCATE TABLE "celida"."execution_run" CASCADE;'))
        db_session.execute(
            text('TRUNCATE TABLE "celida"."population_intervention_pair" CASCADE;')
        )
        db_session.execute(text('TRUNCATE TABLE "celida"."criterion" CASCADE;'))
        db_session.commit()
