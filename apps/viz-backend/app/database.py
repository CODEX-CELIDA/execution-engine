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
        # "check_same_thread": False
        # "keepalives": 1,
        # "keepalives_idle": 30,
        # "keepalives_interval": 10,
        # "keepalives_count": 5,
    },
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
