import logging
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import and_, func
from sqlalchemy.sql import Insert, Select

from execution_engine.omop.db import (  # noqa: F401 -- do not remove (cdm, result) - needed for metadata to work
    base,
    cdm,
    result,
)

from .concepts import Concept


class OMOPSQLClient:
    """A client for the OMOP SQL database.

    This class provides a high-level interface to the OMOP SQL database.

    Parameters
    ----------
    user : str
        The user name to connect to the database.
    password : str
        The password to connect to the database.
    host : str
        The host name of the database.
    port : int
        The port of the database.
    database : str
        The name of the database.
    schema : str
        The name of the schema.
    vocabulary_logger : logging.Logger, optional
        The logger to use for logging vocabulary queries, by default None.
    """

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        schema: str,
        vocabulary_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the OMOP SQL client."""

        self._schema = schema
        connection_string = (
            f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        )

        self._engine = sqlalchemy.create_engine(
            connection_string,
            connect_args={"options": "-csearch_path={}".format(schema)},
            future=True,
        )

        self._metadata = base.metadata
        self._metadata.bind = self._engine

        self._init_result_tables()

        if vocabulary_logger is None:
            vocabulary_logger = logging.getLogger("vocabulary")

            if not vocabulary_logger.hasHandlers():
                vocabulary_logger.setLevel(logging.DEBUG)
                vocabulary_logger.addHandler(logging.NullHandler())
                vocabulary_logger.propagate = False  # Do not propagate to root logger

        self._vocabulary_logger = vocabulary_logger

    def _init_result_tables(self, schema: str = "celida") -> None:
        """
        Initialize the result schema.
        """
        with self.begin() as con:
            if not con.dialect.has_schema(con, schema):
                con.execute(sqlalchemy.schema.CreateSchema(schema))

            result_tables = [
                self._metadata.tables[t]
                for t in self._metadata.tables
                if t.startswith("celida.")
            ]
            assert len(result_tables) > 0, "No results tables found"

            self._metadata.create_all(tables=result_tables, bind=con)

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

        df = self.query(query, code=str(code), vocabulary_id=vocabulary_id)

        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

        return self.get_concept_info(df.iloc[0].ingredient_concept_id)

    def drug_vocabulary_to_ingredient(
        self, vocabulary_id: str, code: str
    ) -> Concept | None:
        """
        Get the ingredient concept for the given drug code in the given vocabulary.
        """
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
        if with_unit:
            query = """
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
            query = """
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
