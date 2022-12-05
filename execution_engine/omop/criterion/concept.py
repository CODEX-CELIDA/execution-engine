from datetime import datetime
from typing import cast

from sqlalchemy import literal_column, table
from sqlalchemy.sql import Insert, TableClause

from ...util import Value
from ..concepts import Concept
from .abstract import Criterion


class ConceptCriterion(Criterion):
    """
    Abstract class for a criterion based on an OMOP concept and optional value.

    This class is not meant to be instantiated directly. Instead, use one of the subclasses.
    Subclasses need to set _OMOP_TABLE and _OMOP_COLUMN_PREFIX (e.g. "visit_occurrence" and "visit").
    These identify the base table in OMOP CDM and the prefix of the concept_id column (e.g. "visit_concept_id").

    """

    DOMAINS: dict[str, dict[str, str | bool]] = {
        "condition": {
            "table": "condition_occurrence",
            "value_required": False,
            "static": True,
        },
        "device": {
            "table": "device_exposure",
            "value_required": False,
            "static": False,
        },
        # "drug": {'table': "drug_exposure", 'value_required': False}, # has its own class with different logic
        "measurement": {
            "table": "measurement",
            "value_required": True,
            "static": False,
        },
        "observation": {
            "table": "observation",
            "value_required": False,
            "static": False,
        },
        "procedure": {
            "table": "procedure_occurrence",
            "value_required": False,
            "static": False,
        },
        "visit": {"table": "visit_occurrence", "value_required": False, "static": True},
    }

    def __init__(
        self,
        name: str,
        exclude: bool,
        concept: Concept,
        value: Value | None = None,
    ):
        super().__init__(name, exclude)

        self._set_omop_variables_from_domain(concept.domain_id)
        self._concept = concept
        self._value = value
        self._table_in: TableClause
        self._table_out: TableClause
        self._start_datetime: datetime | None = None
        self._end_datetime: datetime | None = None

    def _set_omop_variables_from_domain(self, domain_id: str) -> None:
        """
        Set the OMOP table and column prefix based on the domain ID.
        """
        if domain_id.lower() not in self.DOMAINS:
            raise ValueError(f"Domain {domain_id} not supported")

        domain = self.DOMAINS[domain_id.lower()]

        self._OMOP_TABLE = cast(str, domain["table"])
        self._OMOP_COLUMN_PREFIX = domain_id.lower()
        self._OMOP_VALUE_REQUIRED = cast(bool, domain["value_required"])
        self._static = cast(bool, domain["static"])

    def _sql_generate(self, base_sql: Insert) -> Insert:
        """
        Get the SQL representation of the criterion.
        """
        if self._OMOP_VALUE_REQUIRED and self._value is None:
            raise ValueError(f'Value must be set for "{self._OMOP_TABLE}" criteria')

        table_alias = "".join([x[0] for x in self._OMOP_TABLE.split("_")])

        sql = base_sql.select

        concept_id = literal_column(f"{self._OMOP_COLUMN_PREFIX}_concept_id")
        tbl_join = table(
            self._OMOP_TABLE, literal_column("person_id"), concept_id
        ).alias(table_alias)

        sql = sql.join(
            tbl_join.alias(table_alias),
            tbl_join.c.person_id == self._table_in.c.person_id,
        ).filter(tbl_join.columns.corresponding_column(concept_id) == self._concept.id)

        if self._value is not None:
            sql = sql.filter(self._value.to_sql(table_alias))

        # sql += (
        #    f"INNER JOIN {self._OMOP_TABLE} {table_alias} ON ({table_alias}.person_id = table_in.person_id)\n"
        #    f"WHERE ({self._OMOP_COLUMN_PREFIX}_concept_id = {self._concept.id})\n"
        # )
        #
        # if self._value is not None:
        #    sql += f" AND ({self._value.to_sql(table_alias)})\n"

        base_sql.select = sql

        return base_sql
