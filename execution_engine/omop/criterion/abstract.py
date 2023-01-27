import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, cast

import sqlalchemy
from sqlalchemy import Table, bindparam, distinct, literal_column, select
from sqlalchemy.sql import Select, TableClause

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

    @exclude.setter
    def exclude(self, exclude: bool) -> None:
        """Sets the exclude value."""
        self._exclude = exclude

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
        Get the representation of the criterion.
        """
        return (
            f"{self.type}.{self._category.name}.{self._name}(exclude={self._exclude})"
        )

    def __str__(self) -> str:
        """
        Get the name of the criterion.
        """
        return self.name

    def unique_name(self) -> str:
        """
        Get a unique name for the criterion.
        """
        # fixme: will be difficult in the user interface to understand where this name comes from
        # fixme: can we generate a name that is more readable? Or otherwise link it to the FHIR element it came from?
        s = json.dumps(self.dict(), sort_keys=True)
        hash_ = hashlib.md5(  # nosec (just used for naming, not security related)
            s.encode()
        ).hexdigest()
        return f"{self.name}_{hash_}"

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
    _OMOP_DOMAIN: str
    _static: bool

    _table: Table
    _base_table: Table

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

    def _set_omop_variables_from_domain(self, domain_id: str) -> None:
        """
        Set the OMOP table and column prefix based on the domain ID.
        """
        if domain_id.lower() not in self.DOMAINS:
            raise ValueError(f"Domain {domain_id} not supported")

        domain = self.DOMAINS[domain_id.lower()]

        self._OMOP_DOMAIN = domain_id.title()
        self._OMOP_TABLE = domain["table"]
        self._OMOP_COLUMN_PREFIX = domain_id.lower()
        self._OMOP_VALUE_REQUIRED = cast(bool, domain["value_required"])
        self._static = cast(bool, domain["static"])
        self._table = cast(Base, domain["table"]).__table__.alias(self.table_alias)

    @property
    def table_alias(self) -> str:
        """
        Get a table alias for the OMOP table accessed by this criterion.

        The alias is generated by using the first letter of each word in the table name.
        For example, the alias for the table 'condition_occurrence' is 'co'.
        """
        return "".join([x[0] for x in self._OMOP_TABLE.__tablename__.split("_")])

    @property
    def domain(self) -> str:
        """
        Get the domain of the criterion.
        """
        return self._OMOP_DOMAIN

    def _sql_header(
        self, distinct_person: bool = True, person_id: int | None = None
    ) -> Select:
        """
        Generate the header of the SQL query.
        """

        c = self._table.c.person_id

        if distinct_person:
            c = distinct(c).label("person_id")

        query = select(c).select_from(self._table)

        if person_id is None:
            # join the base table to subset patients
            query = query.join(
                self._base_table,
                self._table.c.person_id == self._base_table.c.person_id,
            )
        else:
            # filter by person_id directly
            query = query.filter(self._table.c.person_id == person_id)

        return query

    @abstractmethod
    def _sql_generate(self, query: Select) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        raise NotImplementedError()

    @abstractmethod
    def _sql_filter_concept(self, query: Select) -> Select:
        """
        Add the where clause to the SQL query.

        This is only the base filter (for concept), not for any values or
        constructed values.
        """
        raise NotImplementedError()

    def sql_generate(self, base_table: TableClause) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        self.set_base_table(base_table)

        sql = self._sql_header()
        sql = self._sql_generate(sql)
        sql = self._insert_datetime(sql)
        sql = self._process_exclude(sql)

        return sql

    def set_base_table(self, base_table: Table) -> None:
        """
        Set the base table for the criterion.

        The base table is the table that does a first selection of patients in order
        not to always select all patients in the different criteria.
        This could be the currently active patients.
        """
        assert isinstance(
            base_table, Table
        ), f"base_table must be a Table, not {type(base_table)}"
        self._base_table = base_table

    def _get_datetime_column(self, table: TableClause) -> sqlalchemy.Column:

        table_name = table.original.name

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

    def _insert_datetime(self, sql: SelectInto) -> SelectInto:
        """
        Insert the start_datetime and end_datetime into the query.
        """
        if self._static:
            return sql

        c_start = self._get_datetime_column(self._table)

        if isinstance(sql, Select):
            sql = sql.filter(
                c_start.between(bindparam("start_datetime"), bindparam("end_datetime"))
            )
        else:
            raise ValueError("sql must be a Select")

        return sql

    def _process_exclude(self, sql: str | Select) -> Select:

        if isinstance(sql, str):
            sql = sqlalchemy.text(sql)
            raise ValueError("should this be ever true?")

        if self._exclude:
            if len(sql.columns) > 1:
                # If there are multiple columns, we need to select person_id
                # from the temporary table (can't handle multiple columns in EXCEPT clause)
                sql = select(literal_column("person_id")).select_from(sql)

            sql = (
                select(literal_column("person_id"))
                .select_from(self._base_table)
                .except_(sql)
            )

        return sql

    def sql_select_data(self, person_id: int | None = None) -> Select:
        """
        Get patient data for this criterion
        """
        query = self._sql_header(distinct_person=False, person_id=person_id)
        query = self._sql_select_data(query)
        query = self._sql_filter_concept(query)
        query = self._insert_datetime(query)

        return query

    @abstractmethod
    def _sql_select_data(self, query: Select) -> Select:
        """
        Get the SQL to select the person_id column from the temporary table generated by executing this criterion.
        """
        raise NotImplementedError()

    @staticmethod
    def sql_insert_into_table(
        query: Select, table: TableClause, temporary: bool = True
    ) -> SelectInto:
        """
        Insert the result of the query into the result table.
        """
        if not isinstance(query, Select):
            raise ValueError("sql must be a Select")

        query = select_into(query, table, temporary=temporary)

        return query

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Criterion":
        """
        Create a criterion from a JSON object.
        """
        raise NotImplementedError()
