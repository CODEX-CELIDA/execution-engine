import datetime
import logging
import os
import warnings

import pandas as pd
import pytest
import sqlalchemy
from pytest_postgresql import factories
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.orm.session import sessionmaker
from tqdm import tqdm

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.omop.db.cdm import Person
from tests import concepts
from tests.fixtures import criteria_defs
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
    create_observation,
    create_visit,
    generate_dataframe,
    random_datetime,
    to_extended,
)

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


@pytest.fixture
def visit_start_date():
    visit_start_date = datetime.datetime(2023, 3, 1)
    return visit_start_date


@pytest.fixture
def visit_end_date():
    visit_end_date = datetime.datetime(2023, 3, 31)
    return visit_end_date


# def insert(table: str, data: Dict) -> Any:
#    """
#    Insert data into a table
#    """
#    columns = ", ".join(data.keys())
#    value_placeholder = ", ".join(["%s"] * len(data))
#    sql = f"INSERT INTO {table} ({columns}) VALUES ({value_placeholder}) RETURNING {table}_id"#

#   cursor.execute(sql, list(data.values()))

#    return cursor.fetchone()[0]


@pytest.fixture
def population_intervention():
    population = {
        "COVID19": concepts.COVID19,
        "VENOUS_THROMBOSIS": concepts.VENOUS_THROMBOSIS,
        "HIT2": concepts.HEPARIN_INDUCED_THROMBOCYTOPENIA_WITH_THROMBOSIS,
        "HEPARIN_ALLERGY": concepts.ALLERGY_HEPARIN,
        "HEPARINOID_ALLERGY": concepts.ALLERGY_HEPARINOID,
        "THROMBOCYTOPENIA": concepts.THROMBOCYTOPENIA,
    }

    interventions = {
        "DALTEPARIN": concepts.DALTEPARIN,
        "ENOXAPARIN": concepts.ENOXAPARIN,
        "NADROPARIN_LOW_WEIGHT": concepts.NADROPARIN,
        "NADROPARIN_HIGH_WEIGHT": concepts.NADROPARIN,
        "CERTOPARIN": concepts.CERTOPARIN,
        "FONDAPARINUX": concepts.FONDAPARINUX,
    }

    return population | interventions


@pytest.fixture
def person_combinations(visit_start_date, visit_end_date, population_intervention):

    df = generate_dataframe(population_intervention)

    # Remove invalid combinations
    idx_invalid = df["NADROPARIN_HIGH_WEIGHT"] & df["NADROPARIN_LOW_WEIGHT"]
    df = df[~idx_invalid].copy()

    warnings.warn("remove me")
    df = df.head()

    return df


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


@pytest.fixture
def criteria_extended(
    insert_criteria, criteria, population_intervention, visit_start_date, visit_end_date
):

    idx_static = criteria["static"]
    criteria.loc[idx_static, "start_datetime"] = pd.to_datetime(visit_start_date)
    criteria.loc[idx_static, "end_datetime"] = pd.to_datetime(visit_end_date)
    df = to_extended(
        criteria[["person_id", "concept", "start_datetime", "end_datetime"]],
        observation_start_date=pd.to_datetime(visit_start_date),
        observation_end_date=pd.to_datetime(visit_end_date),
    )
    df.loc[
        :, [c for c in population_intervention.keys() if c not in df.columns]
    ] = False

    df["p_AntithromboticProphylaxisWithLWMH"] = (
        df["COVID19"]
        & ~df["VENOUS_THROMBOSIS"]
        & ~(
            df["HIT2"]
            | df["HEPARIN_ALLERGY"]
            | df["HEPARINOID_ALLERGY"]
            | df["THROMBOCYTOPENIA"]
        )
    )
    df["p_AntithromboticProphylaxisWithFondaparinux"] = (
        df["COVID19"]
        & ~df["VENOUS_THROMBOSIS"]
        & (
            df["HIT2"]
            | df["HEPARIN_ALLERGY"]
            | df["HEPARINOID_ALLERGY"]
            | df["THROMBOCYTOPENIA"]
        )
    )
    df["p_NoAntithromboticProphylaxis"] = df["COVID19"] & df["VENOUS_THROMBOSIS"]

    df["i_AntithromboticProphylaxisWithLWMH"] = (
        df["DALTEPARIN"]
        | df["ENOXAPARIN"]
        | df["NADROPARIN_LOW_WEIGHT"]
        | df["NADROPARIN_HIGH_WEIGHT"]
        | df["CERTOPARIN"]
    )
    df["i_AntithromboticProphylaxisWithFondaparinux"] = df["FONDAPARINUX"]
    df["i_NoAntithromboticProphylaxis"] = ~(
        df["DALTEPARIN"]
        | df["ENOXAPARIN"]
        | df["NADROPARIN_LOW_WEIGHT"]
        | df["NADROPARIN_HIGH_WEIGHT"]
        | df["CERTOPARIN"]
        | df["FONDAPARINUX"]
    )

    df["p_i_AntithromboticProphylaxisWithLWMH"] = (
        df["p_AntithromboticProphylaxisWithLWMH"]
        & df["i_AntithromboticProphylaxisWithLWMH"]
    )
    df["p_i_AntithromboticProphylaxisWithFondaparinux"] = (
        df["p_AntithromboticProphylaxisWithFondaparinux"]
        & df["i_AntithromboticProphylaxisWithFondaparinux"]
    )
    df["p_i_NoAntithromboticProphylaxis"] = (
        df["p_NoAntithromboticProphylaxis"] & df["i_NoAntithromboticProphylaxis"]
    )

    df["p"] = (
        df["p_AntithromboticProphylaxisWithLWMH"]
        | df["p_AntithromboticProphylaxisWithFondaparinux"]
        | df["p_NoAntithromboticProphylaxis"]
    )
    df["i"] = (
        df["i_AntithromboticProphylaxisWithLWMH"]
        | df["i_AntithromboticProphylaxisWithFondaparinux"]
        | df["i_NoAntithromboticProphylaxis"]
    )

    df["p_i"] = (
        df["p_i_AntithromboticProphylaxisWithLWMH"]
        | df["p_i_AntithromboticProphylaxisWithFondaparinux"]
        | df["p_i_NoAntithromboticProphylaxis"]
    )

    return df


def test_recommendation_15_prophylactic_anticoagulation(
    db_session, criteria_extended, visit_start_date, visit_end_date
):
    import itertools

    from execution_engine.clients import omopdb
    from execution_engine.execution_engine import ExecutionEngine

    base_url = (
        "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
    )
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    )

    start_datetime = visit_start_date - datetime.timedelta(days=3)
    end_datetime = visit_end_date + datetime.timedelta(days=3)

    e = ExecutionEngine(verbose=False)

    print(recommendation_url)
    cdd = e.load_recommendation(base_url + recommendation_url, force_reload=False)

    e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)

    df_result = omopdb.query(
        """
    SELECT * FROM celida.recommendation_result
    WHERE

         criterion_name is null
    """
    )
    df_result["valid_date"] = pd.to_datetime(df_result["valid_date"])
    df_result["name"] = df_result["cohort_category"].map(
        {
            "INTERVENTION": "db_i_",
            "POPULATION": "db_p_",
            "POPULATION_INTERVENTION": "db_p_i_",
        }
    ) + df_result["recommendation_plan_name"].fillna("")

    df_result = df_result.rename(columns={"valid_date": "date"})
    df_result = df_result.pivot_table(
        columns="name",
        index=["person_id", "date"],
        values="recommendation_results_id",
        aggfunc=len,
        fill_value=0,
    ).astype(bool)

    plan_names = [
        "AntithromboticProphylaxisWithLWMH",
        "AntithromboticProphylaxisWithFondaparinux",
        "NoAntithromboticProphylaxis",
    ]

    cols = ["_".join(i) for i in itertools.product(["p", "i", "p_i"], plan_names)]
    cols_db = [
        "_".join(i) for i in itertools.product(["db_p", "db_i", "db_p_i"], plan_names)
    ]

    m = criteria_extended.set_index(["person_id", "date"])[cols]
    m = m.join(df_result)

    m.loc[:, [c for c in cols_db if c not in m.columns]] = False

    for plan in plan_names:
        m[f"p_{plan}_eq"] = m[f"p_{plan}"] == m[f"db_p_{plan}"]
        m[f"i_{plan}_eq"] = m[f"i_{plan}"] == m[f"db_i_{plan}"]
        m[f"p_i_{plan}_eq"] = m[f"p_i_{plan}"] == m[f"db_p_i_{plan}"]
        print(plan)
        print("p", (m[f"p_{plan}_eq"]).all(), m[f"p_{plan}"].sum())
        print("i", (m[f"i_{plan}_eq"]).all(), m[f"i_{plan}"].sum())
        print("pi", (m[f"p_i_{plan}_eq"]).all(), m[f"p_i_{plan}"].sum())

    eq = m[[c for c in m.columns if c.endswith("_eq")]]

    # assert eq.all(axis=1).all()

    peq = eq.groupby("person_id").all()
    assert peq.all(axis=1).all()
