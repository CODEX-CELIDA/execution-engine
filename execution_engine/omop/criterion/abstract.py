import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypedDict, cast

import pandas as pd
import sqlalchemy
from sqlalchemy import (
    ColumnElement,
    Date,
    DateTime,
    Table,
    and_,
    bindparam,
    case,
    func,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql import Select, TableClause
from sqlalchemy.sql.functions import concat

from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.concepts import Concept
from execution_engine.omop.db.celida.tables import (
    IntervalTypeEnum,
    RecommendationResultInterval,
)
from execution_engine.omop.db.omop.tables import (
    Base,
    ConditionOccurrence,
    DeviceExposure,
    DrugExposure,
    Measurement,
    Observation,
    ProcedureOccurrence,
    VisitDetail,
    VisitOccurrence,
)
from execution_engine.omop.serializable import Serializable
from execution_engine.util import TimeRange
from execution_engine.util.sql import SelectInto, select_into

__all__ = ["AbstractCriterion", "Criterion"]

Domain = TypedDict(
    "Domain",
    {
        "table": Type[Base],
        "value_required": bool,
        "static": bool,
    },
)


def column_interval_type(interval_type: IntervalType) -> ColumnElement:
    """
    Get a column element for the interval type.

    This is used to insert the interval type into the result table.


    :param interval_type: The interval type.
    :return: A column element for the interval type.
    """
    return bindparam(
        "interval_type", interval_type, type_=IntervalTypeEnum, unique=True
    ).label("interval_type")


def create_conditional_interval_column(condition: ColumnElement) -> ColumnElement:
    """
    Create a conditional column based on a provided condition.

    :param condition: The condition to evaluate for the case statement.
    :return: A SQLAlchemy Column object.
    """

    return (
        case(
            (condition, column_interval_type(IntervalType.POSITIVE)),
            else_=column_interval_type(IntervalType.NEGATIVE),
        )
        .cast(IntervalTypeEnum)
        .label("interval_type")
    )


class AbstractCriterion(Serializable, ABC):
    """
    Abstract base class for Criterion and CriterionCombination.
    """

    def __init__(self, name: str, exclude: bool, category: CohortCategory) -> None:
        self._id = None
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

    def copy(self) -> "AbstractCriterion":
        """
        Copy the criterion.
        """
        return copy.deepcopy(self)

    def invert_exclude(self, inplace: bool = False) -> "AbstractCriterion":
        """
        Invert the exclude flag.
        """
        if inplace:
            self._exclude = not self._exclude
            return self
        else:
            criterion = self.copy()
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

        The name is based on the JSON representation of the criterion, i.e. multiple objects with the same parameters
        will have the same name (in that sense, the uniqueness is related to the parameters, not the object itself).
        """
        # exclusion is only performed during combination of criteria, and therefore criteria behave the same
        # independent of whether they are excluded or not. this should be reflected in the name (therefore we remove
        # the exclude flag from the name)
        criterion_dict = self.dict()
        criterion_dict.pop("exclude", None)

        s = json.dumps(self.dict(), sort_keys=True)

        hash_ = hashlib.md5(  # nosec (just used for naming, not security related)
            s.encode()
        ).hexdigest()

        exclude_str = "(NOT)" if self.exclude else ""

        return f"{self.name}{exclude_str}_{hash_}"


class Criterion(AbstractCriterion):
    """A criterion in a recommendation."""

    _OMOP_TABLE: Type[Base]
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
    during the recommendation period).
    """

    DOMAINS: dict[str, Domain] = {
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
        "visit": {"table": VisitOccurrence, "value_required": False, "static": False},
        "visit_detail": {
            "table": VisitDetail,
            "value_required": False,
            "static": False,
        },
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

    @abstractmethod
    def _create_query(self) -> Select:
        """
        Get the SQL Select query for data required by this criterion.
        """
        raise NotImplementedError()

    def create_query(self) -> Select:
        """
        Get the SQL Select query for data required by this criterion.
        """

        if self._OMOP_VALUE_REQUIRED and (
            not hasattr(self, "_value") or self._value is None
        ):
            raise ValueError(
                f'Value must be set for "{self._OMOP_TABLE.__tablename__}" criteria'
            )

        query = self._create_query()

        query.description = self.description()

        assert (
            len(query.selected_columns) == 4
        ), "Query must select 4 columns: person_id, interval_start, interval_end, interval_type"

        # assert that the output columns are person_id, interval_start, interval_end, type
        assert set([c.name for c in query.selected_columns]) == set(
            ["person_id", "interval_start", "interval_end", "interval_type"]
        ), "Query must select 4 columns: person_id, interval_start, interval_end, interval_type"

        return query

    def process_result(
        self, df: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Process the result of the SQL query.

        Can be overridden by subclasses to perform additional processing of data returned by the SQL query from
        `create_query`.

        :param df: The result of the SQL query.
        :return: A processed DataFrame.
        """
        return df

    def _sql_header(
        self, distinct_person: bool = True, person_id: int | None = None
    ) -> Select:
        """
        Generate the header of the SQL query.
        """

        c_start = self._get_datetime_column(self._table, "start")
        c_end = self._get_datetime_column(self._table, "end")

        query = select(
            self._table.c.person_id,
            c_start.label("interval_start"),
            c_end.label("interval_end"),
        ).select_from(self._table)

        if person_id is not None:
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
    def _sql_filter_concept(self, query: Select) -> Select:
        """
        Add the where clause to the SQL query.

        This is only the base filter (for concept), not for any values or
        constructed values.
        """
        raise NotImplementedError()

    def DEPRECATEDsql_generate(self, base_table: TableClause) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        self.set_base_table(base_table)

        query = self._sql_header()
        # query = self._sql_generate(query)
        query = self._insert_datetime(query)
        query = self._select_per_day(query)

        query.description = self.description()

        return query

    def set_base_table(self, base_table: Table) -> None:
        """
        Set the base table for the criterion.

        The base table is the table that does a first selection of patients in order
        not to always select all patients in the different criteria.
        This could be the currently active patients.
        """
        """assert isinstance(
            base_table, Table
        ), f"base_table must be a Table, not {type(base_table)}"
        self._base_table = base_table"""
        raise NotImplementedError("implement me - we are not using base table anymore")

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

        # todo: are we still using this function? do we add the observation window to the query as columns?

        if self._static:
            # If this criterion is a static criterion, i.e. one whose value does not change over time,
            # then we don't need to filter by datetime,
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

        if "interval_start" not in query.selected_columns:
            query = query.add_columns(c_start.label("interval_start"))
        if "interval_end" not in query.selected_columns:
            query = query.add_columns(c_end.label("interval_end"))

        if isinstance(query, Select):
            query = query.filter(
                and_(
                    c_start <= bindparam("observation_end_datetime"),
                    c_end >= bindparam("observation_start_datetime"),
                )
            )
        else:
            raise ValueError("sql must be a Select")

        return query

    def _filter_days_with_all_values_valid(
        self, query: Select, sql_value: ColumnElement = None
    ) -> Select:
        if not hasattr(self, "_value") or self._value is None:
            return query

        if sql_value is None:
            sql_value = self._value.to_sql(self._table)

        c_datetime = self._get_datetime_column(self._table, "start")
        c_date = func.date(c_datetime)
        query = query.group_by(self._table.c.person_id, c_date)
        query = query.add_columns(c_date.label("valid_from"), c_date.label("valid_to"))
        query = query.having(func.bool_and(sql_value))

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

        query = self._filter_days_with_all_values_valid(query)

        if "valid_from" not in col_names:
            c_start = self._get_datetime_column(self._table, "start").label(
                "valid_from"
            )
            query = query.add_columns(c_start)
        if "valid_to" not in col_names:
            c_end = self._get_datetime_column(self._table, "end").label("valid_to")
            query = query.add_columns(c_end)

        # Adjust the "valid_from" and "valid_to" columns in the query to not extend the observation start/end dates,
        # which are introduced when executing the query. The reason is that these columns determine the start and end
        # date of the date range that is created (using SQL generate_series) for each person. If the criterion extends
        # the observation period, additional dates will be generated for each person (in the SQL output),
        # which is not desired.
        c_valid_from = query.selected_columns["valid_from"]
        c_valid_to = query.selected_columns["valid_to"]

        query = query.with_only_columns(
            *[
                c
                for c in query.selected_columns
                if c.name not in ["valid_from", "valid_to"]
            ]
        )
        query = query.add_columns(
            func.greatest(
                c_valid_from, bindparam("observation_start_datetime", type_=DateTime)
            ).label("valid_from")
        )
        query = query.add_columns(
            func.least(
                c_valid_to, bindparam("observation_end_datetime", type_=DateTime)
            ).label("valid_to")
        )

        query = query.cte("criterion")

        person_dates = (
            select(
                literal_column("person_id"),
                func.generate_series(
                    func.date_trunc(
                        "day", literal_column("valid_from", type_=DateTime)
                    ),
                    func.date_trunc("day", literal_column("valid_to", type_=DateTime)),
                    func.cast(concat(1, "day"), INTERVAL),
                )
                .cast(Date)
                .label("valid_date"),
            )
            .distinct()
            .select_from(query)
        )

        return person_dates

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
    def sql_insert_into_result_table(query: Select) -> SelectInto:
        """
        Insert the result of the query into the result table.
        """
        if not isinstance(query, Select):
            raise ValueError("query must be a Select or CTE")

        query = select_into(
            query, RecommendationResultInterval.__table__, temporary=False
        )
        query.description = query.select.description

        return query

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Criterion":
        """
        Create a criterion from a JSON object.
        """
        raise NotImplementedError()

    @abstractmethod
    def description(self) -> str:
        """
        Return a description of the criterion.
        """
        raise NotImplementedError()
