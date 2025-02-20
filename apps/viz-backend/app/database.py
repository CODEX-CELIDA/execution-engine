from urllib.parse import quote

from settings import config
from sqlalchemy import MetaData, Table, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry, PoolProxiedConnection

connection_dict = config.omop.model_dump()
connection_dict["user"] = quote(connection_dict["user"])
connection_dict["password"] = quote(connection_dict["password"])

connection_string = (
    "postgresql+psycopg://{user}:{password}@{host}:{port}/{database}".format(
        **connection_dict
    )
)

engine = create_engine(
    connection_string,
    pool_pre_ping=True,
    connect_args={
        "options": "-csearch_path={}".format(config.omop.data_schema),
    },
)


@event.listens_for(engine.pool, "checkout")
def set_timezone(
    dbapi_connection: DBAPIConnection,
    connection_record: ConnectionPoolEntry,
    connection_proxy: PoolProxiedConnection,
) -> None:
    """
    Set the timezone for the database connection.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute(
        "SELECT set_config('TIMEZONE', %(timezone)s, false)",
        {"timezone": "Europe/Berlin"},
    )
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

metadata = MetaData()


def reflect_tables() -> dict[str, Table]:
    """
    Reflect tables and views from the specified schemas.
    Returns dynamically reflected tables.
    """
    result_schema = config.omop.result_schema  # Schema for result views
    data_schema = config.omop.data_schema  # Schema for data tables

    interval_result = Table(
        "interval_result", metadata, autoload_with=engine, schema=result_schema
    )
    full_day_coverage = Table(
        "full_day_coverage", metadata, autoload_with=engine, schema=result_schema
    )
    partial_day_coverage = Table(
        "partial_day_coverage", metadata, autoload_with=engine, schema=result_schema
    )
    criterion = Table("criterion", metadata, autoload_with=engine, schema=result_schema)

    concept = Table("concept", metadata, autoload_with=engine, schema=data_schema)
    person = Table("person", metadata, autoload_with=engine, schema=data_schema)
    measurement = Table(
        "measurement", metadata, autoload_with=engine, schema=data_schema
    )
    visit_occurrence = Table(
        "visit_occurrence", metadata, autoload_with=engine, schema=data_schema
    )
    drug_exposure = Table(
        "drug_exposure", metadata, autoload_with=engine, schema=data_schema
    )
    observation = Table(
        "observation", metadata, autoload_with=engine, schema=data_schema
    )
    condition_occurrence = Table(
        "condition_occurrence", metadata, autoload_with=engine, schema=data_schema
    )
    procedure_occurrence = Table(
        "procedure_occurrence", metadata, autoload_with=engine, schema=data_schema
    )

    return {
        "interval_result": interval_result,
        "full_day_coverage": full_day_coverage,
        "partial_day_coverage": partial_day_coverage,
        "criterion": criterion,
        "concept": concept,
        "person": person,
        "measurement": measurement,
        "visit_occurrence": visit_occurrence,
        "drug_exposure": drug_exposure,
        "observation": observation,
        "condition_occurrence": condition_occurrence,
        "procedure_occurrence": procedure_occurrence,
    }


# Reflect tables and expose them
tables = reflect_tables()
interval_result = tables["interval_result"]
full_day_coverage = tables["full_day_coverage"]
partial_day_coverage = tables["partial_day_coverage"]
criterion = tables["criterion"]

concept = tables["concept"]
person = tables["person"]
measurement = tables["measurement"]
visit_occurrence = tables["visit_occurrence"]
drug_exposure = tables["drug_exposure"]
observation = tables["observation"]
condition_occurrence = tables["condition_occurrence"]
procedure_occurrence = tables["procedure_occurrence"]
