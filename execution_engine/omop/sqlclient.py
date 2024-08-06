import logging
from typing import Any
from urllib.parse import quote

import pandas as pd
import sqlalchemy
from sqlalchemy import and_, bindparam, event, func, select, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql import Insert, Select

from execution_engine.omop.db import (  # noqa: F401 -- do not remove (cdm, result) - needed for metadata to work
    base,
)
from execution_engine.omop.db.celida import (  # noqa: F401 # required for metadata initiate all tables, views and triggers
    tables as celida_tables,
)
from execution_engine.omop.db.celida import (  # noqa: F401 # required for metadata initiate all tables, views and triggers
    triggers as celida_triggers,
)
from execution_engine.omop.db.celida import (  # noqa: F401 # required for metadata initiate all tables, views and triggers
    views as celida_views,
)
from execution_engine.omop.db.omop import tables as omop

from .concepts import Concept


def _disable_database_triggers(
    dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
) -> None:
    """
    Disable triggers including foreign key checks (event listener).

    This function is used to disable triggers including foreign key checks when connecting to the OMOP CDM database.

    :param dbapi_connection: The database connection.
    :param connection_record: The connection record.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("SET session_replication_role = 'replica';")
    cursor.close()


def _enable_database_triggers(
    dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
) -> None:
    """
    Enable triggers including  foreign key checks (event listener).

    This function is used to enable triggers including foreign key checks when connecting to the OMOP CDM database.

    :param dbapi_connection: The database connection.
    :param connection_record: The connection record.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("SET session_replication_role = 'origin';")
    cursor.close()


class OMOPSQLClient:
    """A client for the OMOP SQL database.

    This class provides a high-level interface to the OMOP SQL database.

    :param user: The username to connect to the database.
    :param password: The password to connect to the database.
    :param host: The host name of the database.
    :param port: The port of the database.
    :param database: The name of the database.
    :param schema: The name of the schema.
    :param timezone: The timezone to use for the database connection.
    """

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        data_schema: str,
        result_schema: str,
        timezone: str = "Europe/Berlin",
        disable_triggers: bool = False,
    ) -> None:
        """Initialize the OMOP SQL client."""

        self._timezone = timezone

        self._data_schema = data_schema
        self._result_schema = result_schema

        connection_string = f"postgresql+psycopg://{quote(user)}:{quote(password)}@{host}:{port}/{database}"

        self._engine = sqlalchemy.create_engine(
            connection_string,
            connect_args={"options": "-csearch_path={}".format(self._data_schema)},
            future=True,
        )

        if disable_triggers:
            self.disable_triggers()

        self._sessionmaker = sqlalchemy.orm.sessionmaker(bind=self._engine, future=True)

        self._metadata = base.metadata
        self._metadata.bind = self._engine

        self._setup_events()
        self._vocabulary_logger = self._setup_logger("vocabulary")
        self._query_logger = self._setup_logger("query")

    def init(self) -> None:
        """
        Initialize the schema / tables.
        """
        self._init_tables()
        self.disable_interval_check_trigger()

    def _setup_events(self) -> None:
        """
        Set up events for the database connection.
        """

        @event.listens_for(self._engine, "connect")
        def set_timezone(
            dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
        ) -> None:
            """
            Set the timezone for the database connection.
            """
            cursor = dbapi_connection.cursor()
            cursor.execute(
                "SELECT set_config('TIMEZONE', %(timezone)s, false)",
                {"timezone": self._timezone},
            )
            cursor.close()

    def disable_triggers(self) -> None:
        """
        Disable triggers including foreign key checks.
        """
        event.listen(self._engine, "connect", _disable_database_triggers)

    def enable_triggers(self) -> None:
        """
        Enable triggers including foreign key checks.
        """
        event.remove(self._engine, "connect", _enable_database_triggers)

    def disable_interval_check_trigger(self) -> None:
        """
        Disable the overlapping interval check trigger.
        """
        with self.begin() as con:
            con.execute(
                text(
                    "ALTER TABLE {schema}.result_interval DISABLE TRIGGER trigger_result_interval_before_insert".format(
                        schema=self._result_schema
                    )
                )
            )

    def enable_interval_check_trigger(self) -> None:
        """
        Enable the overlapping interval check trigger.
        """
        with self.begin() as con:
            con.execute(
                text(
                    "ALTER TABLE {schema}.result_interval ENABLE TRIGGER trigger_result_interval_before_insert".format(
                        schema=self._result_schema
                    )
                )
            )

    @staticmethod
    def _setup_logger(name: str) -> logging.Logger:
        """Set up a logger for the given name.

        :param name: The name of the logger.
        :return: The logger.
        """
        logger = logging.getLogger(name)
        logger.propagate = False

        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.NullHandler())

        return logger

    def _init_tables(
        self,
    ) -> None:
        """
        Initialize the result schema.
        """
        with self.begin() as con:
            for schema in [self._data_schema, self._result_schema]:
                if not con.dialect.has_schema(con, schema):
                    con.execute(sqlalchemy.schema.CreateSchema(schema))

            self._metadata.create_all(bind=con)

    def session(self) -> sqlalchemy.orm.Session:
        """
        Get a new session.
        """
        return self._sessionmaker()

    def connect(self) -> sqlalchemy.engine.Connection:
        """
        Get a connection to the OMOP CDM database.
        """
        return self._engine.connect()

    def begin(self) -> sqlalchemy.engine.Connection:
        """
        Begin a new transaction.
        """
        return self._engine.begin()

    @property
    def tables(self) -> dict[str, sqlalchemy.Table]:
        """Return the tables in the OMOP CDM database."""
        return self._metadata.tables

    def query(self, sql: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Run the given SQL query against the OMOP CDM database.
        """
        return pd.read_sql(sql, self._engine, params=kwargs)

    def raw_query(
        self, sql: Any, params: dict | None = None
    ) -> sqlalchemy.engine.Result:
        """
        Run the given SQL query against the OMOP CDM database.
        """
        with self.session() as con:
            data = con.execute(sql, params=params)
        return data

    def compile_query(self, query: Select | Insert, params: dict | None = None) -> str:
        """
        Compile the given query against the OMOP CDM database.
        """
        if params is None:
            params = {}

        return str(
            query.params(params).compile(
                dialect=self._engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
        )

    def log_query(self, query: Select | Insert, params: dict | None = None) -> None:
        """
        Log the given query against the OMOP CDM database.
        """
        self._query_logger.info(self.compile_query(query, params) + "\n")

    def get_concept_info(self, concept_id: int) -> Concept:
        """Get the concept info for the given concept ID."""
        concept = omop.Concept.__table__.alias("c")
        query = concept.select().where(concept.c.concept_id == int(concept_id))
        df = self.query(query)

        assert (
            len(df) == 1
        ), f"Expected exactly one concept for {concept_id}, got {len(df)}"

        c = Concept.from_series(df.iloc[0])

        self._vocabulary_logger.info(
            f'"{c.concept_id}","{c.concept_name}","get_concept_info","",""'
        )

        return c

    def get_drug_concept_info(self, concept_id: int) -> dict:
        """Get the drug concept info for the given concept ID."""
        concept = omop.Concept.__table__.alias("c")
        ds = omop.t_drug_strength.alias("ds")

        query = (
            select(
                concept.c.concept_id,
                concept.c.concept_name,
                concept.c.concept_code,
                concept.c.concept_class_id,
                concept.c.vocabulary_id,
                ds.c.amount_value,
                ds.c.amount_unit_concept_id,
                ds.c.numerator_value,
                ds.c.numerator_unit_concept_id,
                ds.c.denominator_value,
                ds.c.denominator_unit_concept_id,
            )
            .select_from(concept)
            .join(
                ds,
                and_(
                    ds.c.drug_concept_id == concept.c.concept_id,
                ),
            )
            .where(concept.c.concept_id == int(concept_id))
        )

        df = self.query(query)

        assert len(df) == 1, f"Expected 1 Concept, got {len(df)}"

        return df.iloc[0].to_dict()

    def get_concept(
        self,
        vocabulary: str,
        code: str,
        standard: bool = False,
        name: str | None = None,
    ) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        logging.info(f"Requesting standard concept: {vocabulary} #{code}")

        concept = omop.Concept.__table__.alias("c")
        query = concept.select().where(
            and_(
                concept.c.vocabulary_id == vocabulary,
                func.lower(concept.c.concept_code) == code.lower(),
                func.now() >= concept.c.valid_start_date,
                func.now() <= concept.c.valid_end_date,
            )
        )

        if standard:
            query = query.where(concept.c.standard_concept == "S")

        if name is not None:
            query = query.where(concept.c.concept_name == name)

        df = self.query(query)

        if not len(df) == 1:
            raise ValueError(
                f"Expected exactly one concept for {vocabulary}#{code} ({name}), got {len(df)}"
            )

        c = Concept.from_series(df.iloc[0])

        self._vocabulary_logger.info(
            f'"{c.concept_id}","{c.concept_name}","get_concept","",""'
        )

        return c

    def drug_vocabulary_to_ingredient_via_ancestor(
        self, vocabulary_id: str, code: str
    ) -> Concept:
        """
        Get the ingredient concept for the given drug code in the given vocabulary.
        """

        c = omop.Concept.__table__.alias("c")
        ci = omop.Concept.__table__.alias("ci")
        ca = omop.t_concept_ancestor.alias("ca")

        query = (
            select(
                ci.c.concept_id.label("ingredient_concept_id"),
                ci.c.concept_name.label("ingredient_name"),
                ci.c.concept_code.label("ingredient_concept_code"),
            )
            .join(
                ca,
                c.c.concept_id == ca.c.descendant_concept_id,
            )
            .join(
                ci,
                ci.c.concept_id == ca.c.ancestor_concept_id,
            )
            .where(
                and_(
                    c.c.concept_code == bindparam("code"),
                    c.c.domain_id == "Drug",
                    ci.c.concept_class_id == "Ingredient",
                    # ci.c.standard_concept == "S",
                    c.c.vocabulary_id == bindparam("vocabulary_id"),
                    c.c.valid_start_date <= func.now(),
                    c.c.valid_end_date >= func.now(),
                    ci.c.valid_start_date <= func.now(),
                    ci.c.valid_end_date >= func.now(),
                )
            )
        )

        df = self.query(query, code=str(code), vocabulary_id=vocabulary_id)

        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

        return self.get_concept_info(df.iloc[0].ingredient_concept_id)

    def drug_vocabulary_to_ingredient(
        self, vocabulary_id: str, code: str
    ) -> Concept | None:
        """
        Get the ingredient concept for the given drug code in the given vocabulary.
        """
        c = omop.Concept.__table__.alias("c")
        cr = omop.t_concept_relationship.alias("cr")
        ci = omop.Concept.__table__.alias("ci")

        query = (
            select(
                c.c.concept_id.label("drug_concept_id"),
                c.c.concept_name.label("drug_concept_name"),
                c.c.concept_class_id.label("drug_concept_class"),
                c.c.concept_code.label("drug_concept_code"),
                ci.c.concept_id.label("ingredient_concept_id"),
                ci.c.concept_name.label("ingredient_concept_name"),
                ci.c.concept_class_id.label("ingredient_concept_class"),
            )
            .join(
                cr,
                cr.c.concept_id_1 == c.c.concept_id,
            )
            .join(
                ci,
                ci.c.concept_id == cr.c.concept_id_2,
            )
            .where(
                and_(
                    c.c.concept_code == bindparam("code"),
                    cr.c.relationship_id == "Maps to",
                    ci.c.concept_class_id == "Ingredient",
                    c.c.domain_id == "Drug",
                    ci.c.standard_concept == "S",
                    c.c.vocabulary_id == bindparam("vocabulary_id"),
                    c.c.valid_start_date <= func.now(),
                    c.c.valid_end_date >= func.now(),
                    ci.c.valid_start_date <= func.now(),
                    ci.c.valid_end_date >= func.now(),
                )
            )
        )

        df = self.query(query, code=str(code), vocabulary_id=vocabulary_id)

        if len(df) == 0:
            return None
        elif len(df) > 1:
            raise ValueError(
                f"Expected concept for {vocabulary_id} #{code}, got {len(df)}"
            )

        c = df.iloc[0]
        self._vocabulary_logger.info(
            f'"{c.drug_concept_id}","{c.drug_concept_name}","drug_vocabulary_to_ingredient","{c.ingredient_concept_id}","{c.ingredient_concept_name}"'
        )

        return self.get_concept_info(c.ingredient_concept_id)

    def drugs_by_ingredient(self, drug_concept_id: str | int) -> pd.DataFrame:
        """
        Get all drugs that map to the given ingredient concept ID.
        """

        c = omop.Concept.__table__.alias("c")
        cd = omop.Concept.__table__.alias("cd")
        ca = omop.t_concept_ancestor.alias("ca")
        ds = omop.t_drug_strength.alias("ds")
        ds_ingr = omop.t_drug_strength.alias("ds_ingr")

        query = (
            select(
                cd.c.concept_id.label("drug_concept_id"),
                cd.c.concept_name.label("drug_name"),
                cd.c.concept_code.label("drug_concept_code"),
                cd.c.concept_class_id.label("drug_concept_class"),
                c.c.concept_id.label("ingredient_concept_id"),
                c.c.concept_name.label("ingredient_name"),
                c.c.concept_code.label("ingredient_concept_code"),
                ds.c.amount_value,
                ds.c.amount_unit_concept_id,
                ds.c.numerator_value,
                ds.c.numerator_unit_concept_id,
                ds.c.denominator_value,
                ds.c.denominator_unit_concept_id,
            )
            .join(
                ca,
                c.c.concept_id == ca.c.ancestor_concept_id,
            )
            .join(
                cd,
                cd.c.concept_id == ca.c.descendant_concept_id,
            )
            .join(
                ds,
                and_(
                    ds.c.drug_concept_id == cd.c.concept_id,
                    ds.c.ingredient_concept_id == c.c.concept_id,
                ),
            )
            .join(
                ds_ingr,
                ds_ingr.c.drug_concept_id == c.c.concept_id,
            )
            .where(
                and_(
                    c.c.concept_id == bindparam("drug_concept_id"),
                    c.c.domain_id == "Drug",
                    ds.c.amount_unit_concept_id == ds_ingr.c.amount_unit_concept_id,
                    func.now().between(c.c.valid_start_date, c.c.valid_end_date),
                    func.now().between(cd.c.valid_start_date, cd.c.valid_end_date),
                    func.now().between(ds.c.valid_start_date, ds.c.valid_end_date),
                )
            )
        )

        df = self.query(query, drug_concept_id=drug_concept_id)

        for _, c in df.iterrows():
            self._vocabulary_logger.info(
                f'"{c.drug_concept_id}","{c.drug_name}","drugs_by_ingredient","{c.ingredient_concept_id}","{c.ingredient_name}"'
            )

        return df

    def concept_related_to(
        self, ancestor: int, descendant: int, relationship_id: str
    ) -> bool:
        """
        Return true if descendant is related to ancestor by the given relationship.

        Note that the relationship is directed, so this will return false if ancestor is related to descendant.
        """
        a = omop.Concept.__table__.alias("a")
        d = omop.Concept.__table__.alias("d")
        cr = omop.t_concept_relationship.alias("ca")

        query = (
            select(
                cr.c.relationship_id.label("relationship_id"),
                d.c.concept_id.label("concept_id"),
                d.c.concept_name.label("concept_name"),
                d.c.concept_code.label("concept_code"),
                d.c.concept_class_id.label("concept_class_id"),
                d.c.vocabulary_id.label("concept_vocab_id"),
            )
            .select_from(cr)
            .join(a, cr.c.concept_id_1 == a.c.concept_id)
            .join(d, cr.c.concept_id_2 == d.c.concept_id)
            .where(
                and_(
                    a.c.concept_id == bindparam("ancestor"),
                    d.c.concept_id == bindparam("descendant"),
                    cr.c.invalid_reason.is_(None),
                    cr.c.relationship_id == bindparam("relationship_id"),
                )
            )
        )

        df = self.query(
            query,
            ancestor=ancestor,
            descendant=descendant,
            relationship_id=relationship_id,
        )

        for _, c in df.iterrows():
            self._vocabulary_logger.info(
                f'"{c.concept_id}","{c.concept_name}","concept_related_to","{ancestor}",""'
            )

        return len(df) > 0
