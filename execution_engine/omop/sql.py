import logging
from typing import Any
from urllib.parse import quote

import pandas as pd
import sqlalchemy
from sqlalchemy import and_, bindparam, event, func, select
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql import Insert, Select

from execution_engine.omop.db import (  # noqa: F401 -- do not remove (cdm, result) - needed for metadata to work
    base,
)
from execution_engine.omop.db.omop import tables as omop

from .concepts import Concept


class OMOPSQLClient:
    """A client for the OMOP SQL database.

    This class provides a high-level interface to the OMOP SQL database.


    :param user: The user name to connect to the database.
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
        schema: str,
        timezone: str = "Europe/Berlin",
    ) -> None:
        """Initialize the OMOP SQL client."""

        self._timezone = timezone

        self._schema = schema
        connection_string = f"postgresql+psycopg://{quote(user)}:{quote(password)}@{host}:{port}/{database}"

        self._engine = sqlalchemy.create_engine(
            connection_string,
            connect_args={"options": "-csearch_path={}".format(schema)},
            future=True,
        )

        self._sessionmaker = sqlalchemy.orm.sessionmaker(bind=self._engine, future=True)

        self._metadata = base.metadata
        self._metadata.bind = self._engine

        self._setup_events()
        self._init_tables()

        self._vocabulary_logger = self._setup_logger("vocabulary")
        self._query_logger = self._setup_logger("query")

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

        # @event.listens_for(self._engine, "before_cursor_execute")
        # def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        #    conn.info.setdefault("query_start_time", []).append(time.time())
        #    logging.debug("Start Query: %s", statement)

        # @event.listens_for(self._engine, "after_cursor_execute")
        # def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        #    total = time.time() - conn.info["query_start_time"].pop(-1)
        #    logging.debug("Query Complete!")
        #    logging.debug("Total Time: %f", total)

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

    def _init_tables(self, schema: str = "celida") -> None:
        """
        Initialize the result schema.
        """
        with self.begin() as con:
            if not con.dialect.has_schema(con, self._schema):
                con.execute(sqlalchemy.schema.CreateSchema(self._schema))
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
        self._query_logger.info(self.compile_query(query, params))

    def get_concept_info(self, concept_id: int) -> Concept:
        """Get the concept info for the given concept ID."""
        concept = self.tables["cds_cdm.concept"]
        query = concept.select().where(concept.c.concept_id == int(concept_id))
        df = self.query(query)

        assert len(df) == 1, f"Expected 1 Concept, got {len(df)}"

        c = Concept.from_series(df.iloc[0])

        self._vocabulary_logger.info(
            f'"{c.concept_id}","{c.concept_name}","get_concept_info","",""'
        )

        return c

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

        concept = self.tables["cds_cdm.concept"]
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

        assert len(df) == 1, f"Expected 1 Concept, got {len(df)}"

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
        # todo: remove me when it is sure that the sqlalchemy query works
        query = """
        SELECT
            c.concept_id        drug_concept_id,
            c.concept_name      drug_concept_name,
            c.concept_class_id  drug_concept_class,
            c.concept_code      drug_concept_code,
            ci.concept_id       ingredient_concept_id,
            ci.concept_name     ingredient_concept_name,
            ci.concept_class_id ingredient_concept_class
        FROM
            concept c,
            concept_ancestor ca,
            concept ci
        WHERE
            c.concept_code = %(code)s
            AND c.concept_id = ca.ancestor_concept_id
            AND ci.concept_id = ca.descendant_concept_id
            AND ci.concept_class_id         = 'Ingredient'
            AND c.domain_id = 'Drug'
            AND ci.standard_concept = 'S'
            AND c.vocabulary_id = %(vocabulary_id)s
            AND NOW() BETWEEN c.valid_start_date AND c.valid_end_date
            AND NOW() BETWEEN ci.valid_start_date AND ci.valid_end_date
        """

        c = omop.Concept.label("c")
        ci = omop.Concept.label("ci")
        ca = omop.t_concept_ancestor.label("ca")

        query = (
            select(
                c.concept_id.label("drug_concept_id"),
                c.concept_name.label("drug_concept_name"),
                c.concept_class_id.label("drug_concept_class"),
                c.concept_code.label("drug_concept_code"),
                ci.concept_id.label("ingredient_concept_id"),
                ci.concept_name.label("ingredient_concept_name"),
                ci.concept_class_id.label("ingredient_concept_class"),
            )
            .join(
                ca,
                c.concept_id == ca.ancestor_concept_id,
            )
            .join(
                ci,
                ci.concept_id == ca.descendant_concept_id,
            )
            .where(
                and_(
                    c.concept_code == bindparam("code"),
                    c.domain_id == "Drug",
                    ci.concept_class_id == "Ingredient",
                    ci.standard_concept == "S",
                    c.vocabulary_id == bindparam("vocabulary_id"),
                    c.valid_start_date <= func.now(),
                    c.valid_end_date >= func.now(),
                    ci.valid_start_date <= func.now(),
                    ci.valid_end_date >= func.now(),
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
        # todo: remove me when it is sure that the sqlalchemy query works
        query = """
        SELECT
            c.concept_id        drug_concept_id,
            c.concept_name      drug_concept_name,
            c.concept_class_id  drug_concept_class,
            c.concept_code      drug_concept_code,
            ci.concept_id       ingredient_concept_id,
            ci.concept_name     ingredient_concept_name,
            ci.concept_class_id ingredient_concept_class
        FROM
            concept c,
            concept_relationship cr,
            concept ci
        WHERE
            c.concept_code = %(code)s
            AND cr.concept_id_1 = c.concept_id
            AND cr.concept_id_2 = ci.concept_id
            AND cr.relationship_id = 'Maps to'
            AND ci.concept_class_id = 'Ingredient'
            AND c.domain_id = 'Drug'
            AND ci.standard_concept = 'S'
            AND c.vocabulary_id = %(vocabulary_id)s
            AND NOW() BETWEEN c.valid_start_date AND c.valid_end_date
            AND NOW() BETWEEN ci.valid_start_date AND ci.valid_end_date
        """

        c = omop.Concept.label("c")
        cr = omop.t_concept_relationship.label("cr")
        ci = omop.Concept.label("ci")

        query = (
            select(
                c.concept_id.label("drug_concept_id"),
                c.concept_name.label("drug_concept_name"),
                c.concept_class_id.label("drug_concept_class"),
                c.concept_code.label("drug_concept_code"),
                ci.concept_id.label("ingredient_concept_id"),
                ci.concept_name.label("ingredient_concept_name"),
                ci.concept_class_id.label("ingredient_concept_class"),
            )
            .join(
                cr,
                cr.concept_id_1 == c.concept_id,
            )
            .join(
                ci,
                ci.concept_id == cr.concept_id_2,
            )
            .where(
                and_(
                    c.concept_code == bindparam("code"),
                    cr.relationship_id == "Maps to",
                    ci.concept_class_id == "Ingredient",
                    c.domain_id == "Drug",
                    ci.standard_concept == "S",
                    c.vocabulary_id == bindparam("vocabulary_id"),
                    c.valid_start_date <= func.now(),
                    c.valid_end_date >= func.now(),
                    ci.valid_start_date <= func.now(),
                    ci.valid_end_date >= func.now(),
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

    def drugs_by_ingredient(
        self, drug_concept_id: str | int, with_unit: bool = True
    ) -> pd.DataFrame:
        """
        Get all drugs that map to the given ingredient concept ID.
        """

        # todo: remove me when it is sure that the sqlalchemy query works
        if with_unit:
            """
            SELECT
                d.concept_id drug_concept_id,
                d.concept_name drug_name,
                d.concept_code drug_concept_code,
                d.concept_class_id drug_concept_class,
                a.concept_id ingredient_concept_id,
                a.concept_name ingredient_name,
                a.concept_code ingredient_concept_code,
                ds.amount_value,
                ds.amount_unit_concept_id,
                -- au.concept_name amount_unit_name,
                ds.numerator_value,
                ds.numerator_unit_concept_id,
                -- nu.concept_name numerator_unit_name,
                ds.denominator_value,
                ds.denominator_unit_concept_id
                -- du.concept_name denominator_unit_name
            FROM
                concept_ancestor ca
            INNER JOIN concept a ON ca.ancestor_concept_id = a.concept_id
            INNER JOIN concept d ON ca.descendant_concept_id = d.concept_id
            LEFT JOIN drug_strength ds ON (d.concept_id = ds.drug_concept_id AND a.concept_id = ds.ingredient_concept_id AND NOW() BETWEEN ds.valid_start_date AND ds.valid_end_date)
            -- LEFT JOIN concept au ON au.concept_id = ds.amount_unit_concept_id
            -- LEFT JOIN concept nu ON nu.concept_id = ds.numerator_unit_concept_id
            -- LEFT JOIN concept du ON du.concept_id = ds.denominator_unit_concept_id
            WHERE
                NOW() BETWEEN a.valid_start_date AND a.valid_end_date
                AND NOW() BETWEEN d.valid_start_date AND d.valid_end_date
                AND ca.ancestor_concept_id = %(drug_concept_id)s
            """
        else:
            """
            SELECT
                d.concept_id drug_concept_id,
                d.concept_name drug_name,
                d.concept_code drug_concept_code,
                d.concept_class_id drug_concept_class
            FROM
                concept_ancestor ca,
                concept a,
                concept d
            WHERE
                ca.ancestor_concept_id = a.concept_id
                AND ca.descendant_concept_id = d.concept_id
                AND NOW() BETWEEN a.valid_start_date AND a.valid_end_date
                AND NOW() BETWEEN d.valid_start_date AND d.valid_end_date
                AND ca.ancestor_concept_id = %(drug_concept_id)s
            """

        a = omop.Concept.label("a")
        d = omop.Concept.label("d")
        ca = omop.t_concept_ancestor.label("ca")

        query = (
            select(
                d.concept_id.label("drug_concept_id"),
                d.concept_name.label("drug_name"),
                d.concept_code.label("drug_concept_code"),
                d.concept_class_id.label("drug_concept_class"),
            )
            .select_from(ca)
            .join(a, ca.ancestor_concept_id == a.concept_id)
            .join(d, ca.descendant_concept_id == d.concept_id)
            .where(
                and_(
                    func.now() >= a.valid_start_date,
                    func.now() <= a.valid_end_date,
                    func.now() >= d.valid_start_date,
                    func.now() <= d.valid_end_date,
                    ca.ancestor_concept_id == bindparam("drug_concept_id"),
                )
            )
        )
        if with_unit:
            ds = omop.t_drug_strength.label("ds")
            query = query.outerjoin(
                ds,
                and_(
                    d.concept_id == ds.drug_concept_id,
                    a.concept_id == ds.ingredient_concept_id,
                    func.now() >= ds.valid_start_date,
                    func.now() <= ds.valid_end_date,
                ),
            )

            query = query.add_columns(
                a.concept_id.label("ingredient_concept_id"),
                a.concept_name.label("ingredient_name"),
                a.concept_code.label("ingredient_concept_code"),
                ds.amount_value,
                ds.amount_unit_concept_id,
                ds.numerator_value,
                ds.numerator_unit_concept_id,
                ds.denominator_value,
                ds.denominator_unit_concept_id,
            )

        df = self.query(query, drug_concept_id=str(drug_concept_id))

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
        # todo: remove me when it is sure that the sqlalchemy query works
        query = """
            SELECT
              cr.relationship_id   AS relationship_id,
              d.concept_id         AS concept_id,
              d.concept_name       AS concept_name,
              d.concept_code       AS concept_code,
              d.concept_class_id   AS concept_class_id,
              d.vocabulary_id      AS concept_vocab_id
            FROM cds_cdm.concept_relationship AS cr
              JOIN cds_cdm.concept AS a ON cr.concept_id_1 = a.concept_id
              JOIN cds_cdm.concept AS d ON cr.concept_id_2 = d.concept_id
            WHERE
              a.concept_id = %(ancestor)s
              and d.concept_id = %(descendant)s
              and cr.invalid_reason IS null and
              relationship_id = %(relationship_id)s
        """

        a = omop.Concept.label("a")
        d = omop.Concept.label("d")
        cr = omop.t_concept_relationship.label("ca")

        query = (
            select(
                [
                    cr.relationship_id.label("relationship_id"),
                    d.concept_id.label("concept_id"),
                    d.concept_name.label("concept_name"),
                    d.concept_code.label("concept_code"),
                    d.concept_class_id.label("concept_class_id"),
                    d.vocabulary_id.label("concept_vocab_id"),
                ]
            )
            .select_from(cr)
            .join(a, cr.concept_id_1 == a.concept_id)
            .join(d, cr.concept_id_2 == d.concept_id)
            .where(
                and_(
                    a.concept_id == bindparam("ancestor"),
                    d.concept_id == bindparam("descendant"),
                    cr.invalid_reason.is_(None),
                    cr.relationship_id == bindparam("relationship_id"),
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
