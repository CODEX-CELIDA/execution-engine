import copy
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, cast

import sqlalchemy
from sqlalchemy import literal_column, select, table
from sqlalchemy.sql import ClauseElement, Insert, TableClause

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.cdm import (
    Base,
    ConditionOccurrence,
    DeviceExposure,
    DrugExposure,
    Measurement,
    Observation,
    ProcedureOccurrence,
    VisitOccurrence,
)
from execution_engine.util.sql import SelectInto, select_into

__all__ = ["AbstractCriterion", "Criterion"]


class AbstractCriterion(ABC):
    """
    Abstract base class for Criterion and CriterionCombination.
    """

    def __init__(self, name: str, exclude: bool, category: CohortCategory) -> None:
        self._name: str = re.sub(r"[ \t]", "-", name)
        self._exclude: bool = exclude

        assert isinstance(
            category, CohortCategory
        ), f"category must be a CohortCategory, not {type(category)}"

        self._category: CohortCategory = category

    @property
    def exclude(self) -> bool:
        """Return the exclude flag."""
        return self._exclude

    @property
    def category(self) -> CohortCategory:
        """Return the category value."""
        return self._category

    @property
    def type(self) -> str:
        """
        Get the type of the criterion.
        """
        return self.__class__.__name__

    @property
    def name(self) -> str:
        """
        Get the name of the criterion.
        """
        return self.type + "_" + self._name

    def invert_exclude(self, inplace: bool = False) -> "AbstractCriterion":
        """
        Invert the exclude flag.
        """
        if inplace:
            self._exclude = not self._exclude
            return self
        else:
            criterion = copy.deepcopy(self)
            criterion._exclude = not criterion._exclude
            return criterion

    def __repr__(self) -> str:
        """
        Get the respresentation of the criterion.
        """
        return (
            f"{self.type}.{self._category.name}.{self._name}(exclude={self._exclude})"
        )

    def __str__(self) -> str:
        """
        Get the name of the criterion.
        """
        return self.name

    @abstractmethod
    def dict(self) -> dict[str, Any]:
        """
        Get the JSON representation of the criterion.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AbstractCriterion":
        """
        Create a criterion from a JSON object.
        """
        raise NotImplementedError()


class Criterion(AbstractCriterion):
    """A criterion in a cohort definition."""

    _OMOP_TABLE: Base
    _OMOP_COLUMN_PREFIX: str
    _OMOP_VALUE_REQUIRED: bool
    _static: bool

    DOMAINS: dict[str, dict[str, Base | bool]] = {
        "condition": {
            "table": ConditionOccurrence,
            "value_required": False,
            "static": True,
        },
        "device": {
            "table": DeviceExposure,
            "value_required": False,
            "static": False,
        },
        "drug": {"table": DrugExposure, "value_required": False, "static": False},
        "measurement": {
            "table": Measurement,
            "value_required": True,
            "static": False,
        },
        "observation": {
            "table": Observation,
            "value_required": False,
            "static": False,
        },
        "procedure": {
            "table": ProcedureOccurrence,
            "value_required": False,
            "static": False,
        },
        "visit": {"table": VisitOccurrence, "value_required": False, "static": True},
    }

    def __init__(self, name: str, exclude: bool, category: CohortCategory) -> None:
        super().__init__(name=name, exclude=exclude, category=category)
        self._table_in: TableClause | None = None
        self._table_out: TableClause | None = None
        self._table_join: TableClause

    def _set_omop_variables_from_domain(self, domain_id: str) -> None:
        """
        Set the OMOP table and column prefix based on the domain ID.
        """
        if domain_id.lower() not in self.DOMAINS:
            raise ValueError(f"Domain {domain_id} not supported")

        domain = self.DOMAINS[domain_id.lower()]

        self._OMOP_TABLE = domain["table"]
        self._OMOP_COLUMN_PREFIX = domain_id.lower()
        self._OMOP_VALUE_REQUIRED = cast(bool, domain["value_required"])
        self._static = cast(bool, domain["static"])
        self._table_join = cast(Base, domain["table"]).__table__.alias(self.table_alias)

    @property
    def table_in(self) -> TableClause:
        """
        Get the name of the table to use as input for this criterion.
        """
        if self._table_in is None:
            raise ValueError("table_in has not been set - call sql_generate first")
        return self._table_in

    @property
    def table_out(self) -> TableClause:
        """
        Get the name of the temporary table generated by executing this criterion.
        """
        if self._table_out is None:
            raise ValueError("table_out has not been set - call sql_generate first")
        return self._table_out

    @property
    def table_alias(self) -> str:
        """
        Get a table alias for the OMOP table accessed by this criterion.

        The alias is generated by using the first letter of each word in the table name.
        For example, the alias for the table 'condition_occurrence' is 'co'.
        """
        return "".join([x[0] for x in self._OMOP_TABLE.__tablename__.split("_")])

    def _sql_header(self, table_in: TableClause, table_out: TableClause) -> Insert:
        """
        Generate the header of the SQL query.
        """
        if table_in is None or table_out is None:
            raise ValueError("table_in and table_out must be set")

        if type(table_in) == type(Base):
            self._table_in = table_in.__table__
        else:
            self._table_in = table_in

        if type(table_out) is str:
            self._table_out = table(table_out, literal_column("person_id")).alias(
                "table_out"
            )
        else:
            self._table_out = table_out

        sel = select_into(
            select(self._table_in.c.person_id).select_from(self._table_in).distinct(),
            into=self._table_out.original,
            temporary=True,
        )

        # from execution_engine.omop.db.result import RecommendationPersonDatum, RecommendationResult
        # sql_insert = insert(RecommendationPersonDatum).from_select(sel.select.columns, sel.select)

        return sel

    @abstractmethod
    def _sql_generate(self, base_sql: SelectInto) -> SelectInto:
        """
        Get the SQL representation of the criterion.
        """
        raise NotImplementedError()

    def sql_generate(
        self,
        table_in: str | None,
        table_out: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> ClauseElement:
        """
        Get the SQL representation of the criterion.
        """
        sql = self._sql_header(table_in, table_out)
        sql = self._sql_generate(sql)
        sql = self._insert_datetime(sql, start_datetime, end_datetime)
        sql = self._sql_post_process(sql)

        return sql

    def _get_datetime_column(self, table: TableClause) -> sqlalchemy.Column:

        table_name = table.original.concept_name

        candidate_prefixes = [
            f"{self._OMOP_COLUMN_PREFIX}_start",
            f"{self._OMOP_COLUMN_PREFIX}",
            f"{table_name}_start",
            f"{table_name}",
        ]
        try:
            column_prefix = next(
                x for x in candidate_prefixes if f"{x}_datetime" in table.columns
            )
        except StopIteration:
            raise ValueError(f"Cannot find datetime column for table {table_name}")

        return table.c[f"{column_prefix}_datetime"]

    def _insert_datetime(
        self, sql: SelectInto, start_datetime: datetime, end_datetime: datetime
    ) -> SelectInto:
        """
        Insert the start_datetime and end_datetime into the query.
        """
        if self._static:
            return sql

        c_start = self._get_datetime_column(self._table_join)

        if isinstance(sql, SelectInto):
            sql.select = sql.select.filter(
                c_start.between(start_datetime, end_datetime)
            )
        else:
            raise ValueError("sql must be a SelectInto")
            # sql = sql.filter(c_start.between(start_datetime, end_datetime))

        return sql

    def _sql_post_process(self, sql: str | ClauseElement) -> ClauseElement:

        if isinstance(sql, str):
            sql = sqlalchemy.text(sql)

        if self._exclude:
            if isinstance(sql, SelectInto):
                sql_select = sql.select
            else:
                sql_select = sql

            if len(sql_select.columns) > 1:
                # If there are multiple columns, we need to select person_id
                # from the temporary table
                sql_select = select(literal_column("person_id")).select_from(sql_select)

            query = select(literal_column("person_id")).select_from(self.table_in)

            if isinstance(sql, SelectInto):
                sql.select = query.except_(sql_select)
            else:
                sql = query.except_(sql_select)

        return sql

    def sql_select(self, with_alias: bool = True) -> ClauseElement:
        """
        Get the SQL to select the person_id column from the temporary table generated by executing this criterion.
        """
        if with_alias:
            table_out = self.table_out
        else:
            table_out = self.table_out.original

        sql = select(table_out.c.person_id)

        return self._sql_post_process(sql)

    def sql_cleanup(self) -> ClauseElement:
        """
        Get the SQL to drop the temporary table generated by executing this criterion.
        """
        # fixme: use proper sqlalchemy syntax
        return f'DROP TABLE "{str(self.table_out.original.concept_name)}"'  # nosec - this is actual SQL code (generated)

    @classmethod
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> "Criterion":
        """
        Create a criterion from a JSON object.
        """
        raise NotImplementedError()
