import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, cast

import sqlalchemy
from sqlalchemy import Table, bindparam, distinct, func, literal_column, or_, select
from sqlalchemy.sql import Select, TableClause
from sqlalchemy.sql.selectable import CTE

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.db.base import Date, DateTime
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
    """
    indicates the value of this concept can be considered constant during a health care encounter (e.g. weight, height,
    allergies, etc.) or if it is subject to change (e.g. laboratory values, vital signs, conditions, etc.)
    """

    _table: Table
    """
    The OMOP table to use for this criterion.
    """

    _base_table: Table
    """
    The table that is used to pre-filter person_ids (usually a table that includes all person_ids that were active
    during the cohort definition period).
    """

    DOMAINS: dict[str, dict[str, Base | bool]] = {
        "condition": {
            "table": ConditionOccurrence,
            "value_required": False,
            "static": False,
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

        c = self._table.c.person_id.label("person_id")

        # todo: remove if not required
        # if distinct_person:
        #    c = distinct(c).label("person_id")

        query = select(c).select_from(self._table)

        if person_id is None:
            # subset patients by filtering on the base table
            query = query.where(
                self._table.c.person_id.in_(
                    select(self._base_table.c.person_id)
                    .distinct()
                    .select_from(self._base_table)
                )
            )
        else:
            # filter by person_id directly
            query = query.filter(self._table.c.person_id == person_id)

        return query

    @property
    @abstractmethod
    def concept(self) -> Concept:
        """
        Get the concept associated with this Criterion
        """
        raise NotImplementedError()

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

        query = self._sql_header()
        query = self._sql_generate(query)
        query = self._insert_datetime(query)
        query = self._select_per_day(query)
        query = self._process_exclude(query)

        return query

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

    def _get_datetime_column(
        self, table: TableClause, type_: str = "start"
    ) -> sqlalchemy.Column:

        table_name = table.original.name

        candidate_prefixes = [
            f"{self._OMOP_COLUMN_PREFIX}_{type_}",
            f"{self._OMOP_COLUMN_PREFIX}",
            f"{table_name}_{type_}",
            f"{table_name}",
        ]
        try:
            column_prefix = next(
                x for x in candidate_prefixes if f"{x}_datetime" in table.columns
            )
        except StopIteration:
            raise ValueError(f"Cannot find datetime column for table {table_name}")

        return table.c[f"{column_prefix}_datetime"]

    def _insert_datetime(self, query: SelectInto) -> SelectInto:
        """
        Insert a WHERE clause into the query to select only entries between the observation start and end datetimes.
        """
        if self._static:
            # If this criterion is a static criterion, i.e. one whose value does not change over time, then we don't
            # need to filter by datetime
            # but we need to add the observation range as the valid range

            query = query.add_columns(
                bindparam("observation_start_datetime", type_=DateTime).label(
                    "valid_from"
                ),
                bindparam("observation_end_datetime", type_=DateTime).label("valid_to"),
            )

            return query

        c_start = self._get_datetime_column(self._table)
        c_end = self._get_datetime_column(self._table, "end")

        if isinstance(query, Select):
            query = query.filter(
                or_(
                    c_start.between(
                        bindparam("observation_start_datetime"),
                        bindparam("observation_end_datetime"),
                    ),
                    c_end.between(
                        bindparam("observation_start_datetime"),
                        bindparam("observation_end_datetime"),
                    ),
                )
            )
        else:
            raise ValueError("sql must be a Select")

        return query

    def _select_per_day(self, query: Select) -> Select:
        """
        Returns a modified Select query that returns a single row for each day between observation start and end date
        on which the criterion (i.e. the select query) is valid/fulfilled.

        The function adds columns for `valid_from` and `valid_to` to the input query if they are not already present.
        It then creates a Common Table Expression (CTE) named 'criterion' from the modified query.

        The function generates a Select query for each person and their valid dates by using the `generate_series`
        function to generate a list of dates between the `valid_from` and `valid_to` dates for each person.
        The resulting query includes columns for `person_id` and `valid_date`, where `valid_date` is a date object that
        represents each day between the `valid_from` and `valid_to` dates for each person.

        :param query: A Select query object representing the original query.
        :return: A modified Select query object that generates a list of person dates.
        """
        col_names = [c.name for c in query.selected_columns]
        if "valid_from" not in col_names:
            c_start = self._get_datetime_column(self._table, "start").label(
                "valid_from"
            )
            query = query.add_columns(c_start)
        if "valid_to" not in col_names:
            c_end = self._get_datetime_column(self._table, "end").label("valid_to")
            query = query.add_columns(c_end)

        query = query.cte("criterion")

        person_dates = (
            select(
                literal_column("person_id"),
                func.generate_series(
                    func.date_trunc("day", literal_column("valid_from")),
                    func.date_trunc("day", literal_column("valid_to")),
                    "1 day",
                )
                .cast(Date)
                .label("valid_date"),
            )
            .distinct()
            .select_from(query)
        )

        return person_dates

    def _process_exclude(self, query: str | Select) -> Select:
        """
        Converts a query, which returns a list of dates (one per row) per person_id, to a query that return a list of
        all days (per person_id) that are within the time range given by observation_start_datetime
        and observation_end_datetime but that are not included in the result of the original query.

        I.e. it performs the following set operation:
        set({day | observation_start_datetime <= day <= observation_end_datetime}) - set(days_from_original_query}
        """

        assert isinstance(query, Select | CTE), "query must be instance of Select"

        if self._exclude:
            distinct_persons = select(
                distinct(self._base_table.c.person_id).label("person_id")
            ).cte("distinct_persons")
            fixed_date_range = (
                select(
                    distinct_persons.c.person_id,
                    func.generate_series(
                        bindparam("observation_start_datetime", type_=DateTime).cast(
                            Date
                        ),
                        bindparam("observation_end_datetime", type_=DateTime).cast(
                            Date
                        ),
                        "1 day",
                    )
                    .cast(Date)
                    .label("valid_date"),
                )
                .select_from(distinct_persons)
                .cte("fixed_date_range")
            )

            query = query.cte("person_dates")

            query = (
                select(fixed_date_range.c.person_id, fixed_date_range.c.valid_date)
                .select_from(
                    fixed_date_range.outerjoin(
                        query,
                        (fixed_date_range.c.person_id == query.c.person_id)
                        & (fixed_date_range.c.valid_date == query.c.valid_date),
                    )
                )
                .where(query.c.valid_date.is_(None))
                .order_by(fixed_date_range.c.person_id, fixed_date_range.c.valid_date)
            )

        return query

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
            raise ValueError("query must be a Select or CTE")

        query = select_into(query, table, temporary=temporary)

        return query

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Criterion":
        """
        Create a criterion from a JSON object.
        """
        raise NotImplementedError()
