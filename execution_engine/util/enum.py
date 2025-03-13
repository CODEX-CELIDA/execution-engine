from enum import StrEnum
from typing import Any

from sqlalchemy import NUMERIC, ColumnElement
from sqlalchemy import Interval as SQLInterval
from sqlalchemy import func
from sqlalchemy.sql.functions import concat

from execution_engine.util.value import ucum_to_postgres
from execution_engine.util.value.value import ValueNumeric


class TimeUnit(StrEnum):
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

    def __repr__(self) -> str:
        """
        Get the string representation of the category.
        """
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        """
        Get the string representation of the category.
        """
        return self.name

    def __rmul__(self, other: Any) -> Any:
        """
        Multiply a number by a TimeUnit.

        :param other: The number to multiply.
        :return: A ValueNumeric object.
        """
        if isinstance(other, float) and other.is_integer() or isinstance(other, int):
            return ValueNumeric[int, TimeUnit](value=other, unit=self)
        else:
            return ValueNumeric[float, TimeUnit](value=other, unit=self)

    def to_sql_interval(self) -> ColumnElement:
        """
        Get the SQL Interval representation of the value.

        E.g. TimeUnit.DAY corresponds to SQL "INTERVAL '1 DAY'"
        """
        return func.cast(concat(1, ucum_to_postgres[self.value]), SQLInterval)

    def to_sql_interval_length_seconds(self) -> ColumnElement:
        """
        Get the SQL Interval representation of the value in seconds.

        E.g. ValuePeriod(value=1, unit=TimeUnit.DAY) corresponds to 86400 seconds.
        """
        return func.cast(func.extract("EPOCH", self.to_sql_interval()), NUMERIC).label(
            "duration_seconds"
        )


class TimeIntervalType(StrEnum):
    """
    Types of time intervals to aggregate criteria over.
    """

    MORNING_SHIFT = "morning_shift"
    AFTERNOON_SHIFT = "afternoon_shift"
    NIGHT_SHIFT = "night_shift"
    NIGHT_SHIFT_BEFORE_MIDNIGHT = "night_shift_before_midnight"
    NIGHT_SHIFT_AFTER_MIDNIGHT = "night_shift_after_midnight"
    DAY = "day"
    ANY_TIME = "any_time"

    def __repr__(self) -> str:
        """
        Get the string representation of the category.
        """
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        """
        Get the string representation of the category.
        """
        return self.name
