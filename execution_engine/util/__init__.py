from abc import ABC, ABCMeta, abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import pendulum
from portion import Interval as IntervalType
from portion import closed as interval_closed
from pydantic import BaseModel, PositiveInt, root_validator, validator
from sqlalchemy import TableClause, and_, func, literal_column
from sqlalchemy.sql.elements import (
    BinaryExpression,
    ClauseList,
    ColumnClause,
    ColumnElement,
)

from execution_engine.omop.concepts import Concept

ucum_to_postgres = {
    "s": "second",
    "min": "minute",
    "h": "hour",
    "d": "day",
    "wk": "week",
    "mo": "month",
    "a": "year",
}


def get_precision(value: float | int) -> int:
    """
    Get the precision (i.e. the number of decimal places) of a float or int.
    """
    if not isinstance(value, (int, float)):
        raise TypeError("value must be a float or an int.")

    n = str(value)

    if "." not in n:
        if "e" in n:
            return int(
                n.split("e")[1][1:]
            )  # In case of scientific notation, return the magnitude as precision.
        return 0  # the number is an integer, so its precision is 0
    return len(
        n.split(".")[1]
    )  # the number of digits after the decimal point is the precision


class Value(BaseModel, ABC):
    """A value in a criterion."""

    @staticmethod
    def _get_column(
        table: TableClause | None, column_name: str | ColumnElement
    ) -> ColumnClause:
        if table is not None and isinstance(column_name, ColumnElement):
            raise ValueError(
                "If table is set, column_name must be a string, not a ColumnClause."
            )

        if table is not None:
            return table.c[column_name]

        if isinstance(column_name, ColumnElement):
            return column_name

        raise ValueError(
            "column_name must be a string and table set, or a ColumnClause."
        )

    @abstractmethod
    def to_sql(
        self,
        table: TableClause | None,
        column_name: str = "value_as_number",
        with_unit: bool = True,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.
        """

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Get the JSON representation of the value.
        """
        return {
            "class_name": self.__class__.__name__,
            "data": super().dict(*args, **kwargs),
        }


class ValueNumber(Value):
    """
    A value of type number.
    """

    unit: Concept
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None

    @root_validator  # type: ignore
    def validate_value(cls, values: dict) -> dict:
        """
        Validate that value or value_min/value_max is set.
        """

        if values.get("value") is None:
            if values.get("value_min") is None and values.get("value_max") is None:
                raise ValueError("Either value or value_min and value_max must be set.")
            if (
                values.get("value_min") is not None
                and values.get("value_max") is not None
            ):
                if values.get("value_min") > values.get("value_max"):  # type: ignore
                    raise ValueError(
                        "value_min must be less than or equal to value_max."
                    )
        else:
            if (
                values.get("value_min") is not None
                or values.get("value_max") is not None
            ):
                raise ValueError(
                    "value and value_min/value_max are mutually exclusive."
                )
        return values

    def __str__(self) -> str:
        """
        Get the string representation of the value.
        """
        if self.value is not None:
            return f"Value == {self.value} {self.unit.concept_name}"
        elif self.value_min is not None and self.value_max is not None:
            return f"{self.value_min} <= Value <= {self.value_max} {self.unit.concept_name}"
        elif self.value_min is not None:
            return f"Value >= {self.value_min} {self.unit.concept_name}"
        elif self.value_max is not None:
            return f"Value <= {self.value_max} {self.unit.concept_name}"

        raise ValueError("Value is not set.")

    def __repr__(self) -> str:
        """
        Get the string representation of the value.
        """
        return str(self)

    @classmethod
    def parse(cls, s: str, unit: Concept) -> "ValueNumber":
        """
        Parse a string representation of a value.
        """

        value_min = None
        value_max = None
        value = None

        if s.startswith(">="):
            value_min = float(s[2:])
        elif s.startswith("<="):
            value_max = float(s[2:])
        elif s.startswith(">"):
            raise ValueError("ValueNumber does not support >.")
        elif s.startswith("<"):
            raise ValueError("ValueNumber does not support <.")
        elif "--" in s:
            parts = s.split("--")
            value_min = float(parts[0])
            value_max = -float(parts[1])
        elif "-" in s and s.count("-") > 1:  # Check for more than one '-' sign.
            parts = [
                part for part in s.split("-") if part
            ]  # Split and ignore empty strings
            value_min = -float(parts[0])
            value_max = float(parts[1])
        elif "-" in s and s.count("-") == 1 and not s.startswith("-"):
            parts = s.split("-")
            value_min = float(parts[0])
            value_max = float(parts[1])
        else:
            value = float(s)

        return cls(value_min=value_min, value_max=value_max, value=value, unit=unit)

    def to_sql(
        self,
        table: TableClause | None = None,
        column_name: str | ColumnClause = "value_as_number",
        with_unit: bool = True,
    ) -> ColumnElement:
        """
        Get the sqlalchemy representation of the value.
        """

        clauses = []

        c = self._get_column(table, column_name)

        if with_unit:
            c_unit = self._get_column(table, "unit_concept_id")
            clauses.append(c_unit == self.unit.concept_id)

        def eps(number: float) -> float:
            return min(0.001, 10 ** (-(get_precision(number) + 1)))

        if self.value is not None:
            clauses.append(func.abs(c - self.value) < eps(self.value))
        else:
            if self.value_min is not None:
                clauses.append((c - self.value_min) >= -eps(self.value_min))
            if self.value_max is not None:
                clauses.append((c - self.value_max) <= eps(self.value_max))

        return and_(*clauses)


class ValueConcept(Value):
    """
    A value of type concept.
    """

    value: Concept

    def to_sql(
        self,
        table: TableClause | None,
        column_name: str | ColumnClause = "value_as_concept_id",
        with_unit: bool = False,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.
        """
        if with_unit:
            raise ValueError("ValueConcept does not support units.")

        c = self._get_column(table, column_name)

        clause = c == self.value.concept_id

        return clause

    def __str__(self) -> str:
        """
        Get the string representation of the value.
        """
        return f"Value == {str(self.value)}"

    def __repr__(self) -> str:
        """
        Get the string representation of the value.
        """
        return str(self)


class TimeRange(BaseModel):
    """
    A time range.
    """

    start: datetime
    end: datetime
    name: str | None

    @validator("start", "end", pre=False, each_item=False)
    def check_timezone(cls, v: datetime) -> datetime:
        """
        Check that the start, end parameters are timezone-aware.
        """
        if not v.tzinfo:
            raise ValueError("Datetime object must be timezone-aware")
        return v

    @classmethod
    def from_tuple(
        cls, dt: tuple[datetime | str, datetime | str], name: str | None = None
    ) -> "TimeRange":
        """
        Create a time range from a tuple of datetimes.
        """
        return cls(start=dt[0], end=dt[1], name=name)

    @property
    def period(self) -> pendulum.Period:
        """
        Get the period of the time range.
        """
        return pendulum.period(start=self.start.date(), end=self.end.date())

    def date_range(self) -> set[date]:
        """
        Get the date range of the time range.
        """
        return set(self.period.range("days"))

    @property
    def duration(self) -> timedelta:
        """
        Get the duration of the time range.
        """
        return self.end - self.start

    def interval(self, as_timestamp: bool = True) -> IntervalType:
        """
        Get the interval of the time range.

        :param as_timestamp: If True, return the interval as a timestamp (i.e. the number of seconds).
        :return: The interval.
        """

        if as_timestamp:
            return interval_closed(self.start.timestamp(), self.end.timestamp())
        else:
            return interval_closed(self.start, self.end)

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, datetime]:
        """
        Get the dictionary representation of the time range.
        """
        prefix = self.name + "_" if self.name else ""
        return {
            prefix + "start_datetime": self.start,
            prefix + "end_datetime": self.end,
        }


class Interval(str, Enum):
    """
    An interval of time used in Drug Dosing.
    """

    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"
    DAY = "d"
    WEEK = "wk"
    MONTH = "mo"
    YEAR = "a"


class Dosage(BaseModel):
    """
    A dosage consisting of a dose, frequency and interval.
    """

    dose: ValueNumber
    frequency: PositiveInt
    interval: Interval

    class Config:
        """
        Pydantic configuration.
        """

        use_enum_values = True
        """ Use enum values instead of names. """


class AbstractPrivateMethods(ABCMeta):
    """
    A metaclass that prevents overriding of methods decorated with @typing.final.
    """

    def __new__(mcs, name: str, bases: tuple, class_dict: dict) -> Any:
        """
        Instantiate a new class.

        Checks for __final__ attribute set on methods of parent classes (via @typing.final decorator)
        and raises an error if a child class tries to override them.
        """
        private = {
            key: base.__qualname__
            for base in bases
            for key, value in vars(base).items()
            if callable(value) and getattr(value, "__final__", False)
        }

        if any(key in private for key in class_dict):
            message = ", ".join([f"{v}.{k}" for k, v in private.items()])
            raise RuntimeError(f"Methods {message} may not be overriden")
        return super().__new__(mcs, name, bases, class_dict)


def value_factory(class_name: str, data: dict) -> Value:
    """
    Get a value object from a class name and data.

    :param class_name: The name of the class to instantiate.
    :param data: The data to pass to the class constructor.
    :return: The value object.
    :raises ValueError: If the class name is not recognized.
    """

    class_map = {
        "ValueNumber": ValueNumber,
        "ValueConcept": ValueConcept,
    }

    """Create a value from a dictionary."""
    if class_name not in class_map:
        raise ValueError(f"Unknown value class {class_name}")

    return class_map[class_name](**data)
