from settings import config
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

connection_string = (
    "postgresql+psycopg://{user}:{password}@{host}:{port}/{database}".format(
        **config.omop.dict()
    )
)

engine = create_engine(
    connection_string,
    pool_pre_ping=True,
    connect_args={
        "options": "-csearch_path={}".format(config.omop.db_schema),
    },
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
