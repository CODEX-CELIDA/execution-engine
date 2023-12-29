import logging
import os
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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIMEZONE = "Europe/Berlin"


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

        con.execute(
            text("SET session_replication_role = 'replica';")
        )  # Disable foreign key checks

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

        con.execute(
            text("SET session_replication_role = 'origin';")
        )  # Enable foreign key checks

        con.commit()

    yield sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db_session(db_setup):
    session = db_setup()

    yield session

    session.execute(text('TRUNCATE TABLE "cds_cdm"."person" CASCADE;'))
    session.commit()
