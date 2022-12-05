import copy
import re
from abc import ABC, abstractmethod
from datetime import datetime

import sqlalchemy
from sqlalchemy import TypeDecorator, literal_column, select, table
from sqlalchemy.engine import Dialect
from sqlalchemy.sql import ClauseElement, Insert, TableClause


class AbstractCriterion(ABC):
    """
    Abstract base class for Criterion and CriterionCombination.
    """

    def __init__(self, name: str, exclude: bool = False):
        self._name: str = re.sub(r"[ \t]", "-", name)
        self._exclude: bool = exclude

    @property
    def exclude(self) -> bool:
        """Return the exclude flag."""
        return self._exclude

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
        return str(self)

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

    def __str__(self) -> str:
        """
        Get the name of the criterion.
        """
        return self.type + "_" + self._name


class DateTimeType(TypeDecorator):
    """
    SQLAlchemy type for datetime columns.
    """

    impl = sqlalchemy.types.DateTime

    def process_literal_param(self, value: datetime, dialect: Dialect) -> str:
        """
        Convert a datetime to a string.
        """
        return value.strftime("'%Y-%m-%d %H:%M:%S'")


class Criterion(AbstractCriterion):
    """A criterion in a cohort definition."""

    _OMOP_TABLE: str
    _OMOP_COLUMN_PREFIX: str
    _OMOP_VALUE_REQUIRED: bool
    _static: bool

    def __init__(self, name: str, exclude: bool = False):
        super().__init__(name, exclude)
        self._table_in: TableClause | None = None
        self._table_out: TableClause | None = None

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

    def _sql_header(
        self, table_in: TableClause | str, table_out: TableClause | str
    ) -> Insert:
        """
        Generate the header of the SQL query.
        """
        if table_in is None or table_out is None:
            raise ValueError("table_in and table_out must be set")

        if isinstance(table_in, str):
            self._table_in = table(table_in, literal_column("person_id")).alias(
                "table_in"
            )
        else:
            self._table_in = table_in

        if isinstance(table_out, str):
            self._table_out = table(table_out, literal_column("person_id")).alias(
                "table_out"
            )
        else:
            self._table_out = table_out

        sel = select(self._table_in.c.person_id).select_from(self._table_in).distinct()
        query = self._table_out.original.insert().from_select(
            [self._table_out.c.person_id], sel
        )

        return query

    @abstractmethod
    def _sql_generate(self, sql_header: Insert) -> Insert:
        """
        Get the SQL representation of the criterion.
        """
        raise NotImplementedError()

    def sql_generate(
        self,
        table_in: str | None,
        table_out: str,
        datetime_start: datetime,
        datetime_end: datetime,
    ) -> ClauseElement:
        """
        Get the SQL representation of the criterion.
        """
        sql = self._sql_header(table_in, table_out)
        sql = self._sql_generate(sql)
        sql = self._insert_datetime(sql, datetime_start, datetime_end)
        sql = self._sql_post_process(sql)

        return sql

    def _insert_datetime(
        self, sql: Insert, datetime_start: datetime, datetime_end: datetime
    ) -> Insert:
        """
        Insert the datetime_start and datetime_end into the query.
        """
        if self._static:
            return sql

        c_start = literal_column(f"{self._OMOP_TABLE}_start_datetime", DateTimeType())
        # c_end = literal_column(f"{self._OMOP_TABLE}_end_datetime", DateTimeType())

        if isinstance(sql, Insert):
            sql.select = sql.select.filter(
                c_start.between(datetime_start, datetime_end)
            )
        else:
            sql = sql.filter(c_start.between(datetime_start, datetime_end))

        return sql

    def _sql_post_process(self, sql: str | ClauseElement) -> ClauseElement:

        if isinstance(sql, str):
            sql = sqlalchemy.text(sql)

        query = select(literal_column("person_id")).select_from(self.table_in)
        if self._exclude:
            if isinstance(sql, Insert):
                sql.select = query.except_(sql.select)
            else:
                sql = query.except_(sql)
            # sql = f"(SELECT person_id FROM {self.table_in}) EXCEPT ({sql})"  # nosec - this is actual SQL code (generated)
        return sql

    def sql_select(self) -> str:
        """
        Get the SQL to select the person_id column from the temporary table generated by executing this criterion.
        """
        sql = select(literal_column("person_id")).select_from(self.table_out)
        # sql = f"SELECT person_id FROM {self.table_out}"  # nosec - this is actual SQL code (generated)

        return self._sql_post_process(sql)

    def sql_cleanup(self) -> str:
        """
        Get the SQL to drop the temporary table generated by executing this criterion.
        """
        return f"DROP table {self.table_out}"
