import datetime
import logging
import os

import pandas as pd
import pytest
import sqlalchemy
from pytest_postgresql import factories
from sqlalchemy import create_engine, text
from sqlalchemy.orm.session import sessionmaker
from tqdm import tqdm

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.omop.db.cdm import Person
from tests import concepts
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
    create_observation,
    create_visit,
    random_datetime,
)
from tests.parameter import criteria_defs

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

postgresql_in_docker = factories.postgresql_noproc()
postgresql = factories.postgresql("postgresql_in_docker")


@pytest.fixture
def db_session(postgresql):
    """Session for SQLAlchemy."""
    pg_host = postgresql.info.host
    pg_port = postgresql.info.port
    pg_user = postgresql.info.user
    pg_password = postgresql.info.password
    pg_db = postgresql.info.dbname

    os.environ["OMOP__USER"] = pg_user
    os.environ["OMOP__PASSWORD"] = pg_password
    os.environ["OMOP__HOST"] = pg_host
    os.environ["OMOP__PORT"] = str(pg_port)
    os.environ["OMOP__DATABASE"] = pg_db
    os.environ["OMOP__SCHEMA"] = "cds_cdm"

    # with DatabaseJanitor(
    #    pg_user, pg_host, pg_port, pg_db, postgresql.info.server_version, pg_password
    # ):
    connection_str = (
        f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    )
    engine = create_engine(connection_str)
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
            df = pd.read_csv(f"tests/omop_cdm/{table}.csv.gz")
            df.to_sql(table, con, schema="cds_cdm", if_exists="append", index=False)

        con.commit()

        logger.info("yielding a sessionmaker against the test postgres db.")

        yield sessionmaker(bind=engine, expire_on_commit=False)

        # metadata.drop_all(con)


@pytest.fixture
def criteria(
    person_combinations, visit_start_date, visit_end_date, population_intervention
):

    entries = []

    for person_id, row in tqdm(
        person_combinations.iterrows(),
        total=len(person_combinations),
        desc="Generating criteria",
    ):

        for criterion in population_intervention:

            if not row[criterion]:
                continue

            params = criteria_defs[criterion]

            entry = {
                "person_id": person_id,
                "type": params["type"],
                "concept": criterion,
                "concept_id": population_intervention[criterion],
                "static": params["static"],
            }

            if params["type"] == "condition":
                entry["start_datetime"] = random_datetime(visit_start_date)
                entry["end_datetime"] = random_datetime(visit_end_date)
            elif params["type"] == "observation":
                entry["start_datetime"] = datetime.datetime(2023, 3, 15, 12, 0, 0)
            elif params["type"] == "drug":
                entry["start_datetime"] = datetime.datetime(2023, 3, 2, 12, 0, 0)
                entry["end_datetime"] = datetime.datetime(2023, 3, 3, 12, 0, 0)
                entry["quantity"] = (
                    params["dosage_threshold"] - 1
                    if "dosage_threshold" in params
                    else params["dosage"]
                )
                entry["quantity"] *= 2  # over two days
            # create_measurement(vo, concepts.LAB_APTT, datetime.datetime(2023,3,4, 18), 50, concepts.UNIT_SECOND)

            else:
                raise NotImplementedError()
            entries.append(entry)

        if row["NADROPARIN_HIGH_WEIGHT"] or row["NADROPARIN_LOW_WEIGHT"]:
            entry = {
                "person_id": person_id,
                "type": "measurement",
                "concept": "WEIGHT",
                "concept_id": concepts.WEIGHT,
                "start_datetime": datetime.datetime.combine(
                    visit_start_date, datetime.time()
                )
                + datetime.timedelta(days=1),
                "value": 71 if row["NADROPARIN_HIGH_WEIGHT"] else 69,
                "unit_concept_id": concepts.UNIT_KG,
                "static": True,
            }
            entries.append(entry)

    dfe = pd.DataFrame(entries)

    return dfe


@pytest.fixture
def insert_criteria(db_session, criteria, visit_start_date, visit_end_date):
    session = db_session()

    session.execute(
        text("SET session_replication_role = 'replica';")
    )  # Disable foreign key checks

    for person_id, g in tqdm(
        criteria.groupby("person_id"),
        total=criteria["person_id"].nunique(),
        desc="Inserting criteria",
    ):

        p = Person(
            person_id=person_id,
            gender_concept_id=0,
            year_of_birth=1990,
            month_of_birth=1,
            day_of_birth=1,
            race_concept_id=0,
            ethnicity_concept_id=0,
        )
        vo = create_visit(p, visit_start_date, visit_end_date)

        person_entries = [p, vo]

        for _, row in g.iterrows():

            if row["type"] == "condition":
                entry = create_condition(vo, row["concept_id"])
            elif row["type"] == "observation":
                entry = create_observation(
                    vo, row["concept_id"], datetime=row["start_datetime"]
                )
            elif row["type"] == "measurement":
                entry = create_measurement(
                    vo,
                    measurement_concept_id=row["concept_id"],
                    datetime=row["start_datetime"],
                    value_as_number=row["value"],
                    unit_concept_id=row["unit_concept_id"],
                )
            elif row["type"] == "drug":
                entry = create_drug_exposure(
                    vo=vo,
                    drug_concept_id=row["concept_id"],
                    start_datetime=row["start_datetime"],
                    end_datetime=row["end_datetime"],
                    quantity=row["quantity"],
                )
            # create_measurement(vo, concepts.LAB_APTT, datetime.datetime(2023,3,4, 18), 50, concepts.UNIT_SECOND)

            else:
                raise NotImplementedError()

            person_entries.append(entry)

        session.add_all(person_entries)
        session.commit()
