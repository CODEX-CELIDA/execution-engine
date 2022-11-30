import logging
from typing import Any

import pandas as pd
import psycopg2  # noqa: F401 -- do not remove - needed for sqlalchemy to work
import sqlalchemy

from .concepts import Concept


class OMOPSQLClient:
    """A client for the OMOP CDM database."""

    def __init__(
        self, user: str, password: str, host: str, port: int, database: str, schema: str
    ) -> None:
        """Initialize the OMOP SQL client."""
        self._engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
            connect_args={"options": "-csearch_path={}".format(schema)},
        )

    def query(self, sql: str, **kwargs: str | int) -> pd.DataFrame:
        """
        Run the given SQL query against the OMOP CDM database.
        """
        return pd.read_sql(sql, self._engine, params=kwargs)

    def get_concept_info(self, concept_id: str) -> Concept:
        """Get the concept info for the given concept ID."""
        query = "SELECT * FROM concept WHERE concept_id = %(concept_id)s"
        df = self.query(query, concept_id=str(concept_id))

        assert len(df) == 1, f"Expected 1 result, got {len(df)}"

        return Concept.from_series(df.iloc[0])

    def get_concept(
        self, vocabulary: str, code: str, standard: bool = False
    ) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        logging.info(f"Requesting standard concept: {vocabulary} #{code}")
        query = """
        SELECT *
        FROM concept
        WHERE vocabulary_id = %(vocabulary)s
            AND concept_code = %(code)s
            AND NOW() BETWEEN valid_start_date AND valid_end_date
        """
        if standard:
            query += "AND standard_concept = 'S'"

        df = self.query(query, vocabulary=vocabulary, code=code)

        assert len(df) == 1, f"Expected 1 result, got {len(df)}"

        return Concept.from_series(df.iloc[0])

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

    def drug_vocabulary_to_ingredient(self, vocabulary_id: str, code: str) -> Concept:
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

        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

        return self.get_concept_info(df.iloc[0].ingredient_concept_id)

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

        return df
