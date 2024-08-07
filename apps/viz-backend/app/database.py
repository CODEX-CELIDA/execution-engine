from urllib.parse import quote

from settings import config
from sqlalchemy import create_engine, event
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
