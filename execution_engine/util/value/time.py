from pydantic import validator
from pydantic.types import NonNegativeInt
from sqlalchemy import ColumnElement
from sqlalchemy import Interval as SQLInterval
from sqlalchemy import TableClause
from sqlalchemy.sql.functions import concat, func

from execution_engine.util.enum import TimeUnit
from execution_engine.util.value.value import Value, ValueNumeric, check_int

_ucum_to_postgres = {
    "s": "second",
    "min": "minute",
    "h": "hour",
    "d": "day",
    "wk": "week",
    "mo": "month",
    "a": "year",
}


class ValuePeriod(ValueNumeric[NonNegativeInt, TimeUnit]):
    """
    A non-negative integer value with a unit of type TimeUnit, where value_min and value_max are not allowed.
    """

    value_min: None = None
    value_max: None = None
    _validate_value = validator("value", pre=True, allow_reuse=True)(check_int)

    def __str__(self) -> str:
        """
        Get the string representation of the value.
        """
        if self.value == 1:
            return f"{self.unit}"

        return f"{self.value} {self.unit}"

    def to_sql_interval(self) -> ColumnElement:
        """
        Get the SQL Interval representation of the value.

        E.g. ValuePeriod(value=1, unit=TimeUnit.DAY) corresponds to SQL "INTERVAL '1 DAY'"
        """
        return func.cast(
            concat(self.value, _ucum_to_postgres[str(self.unit)]), SQLInterval
        )


class ValueCount(ValueNumeric[NonNegativeInt, None]):
    """
    A non-negative integer value without a unit.
    """

    unit: None = None

    _validate_value = validator("value", pre=True, allow_reuse=True)(check_int)
    _validate_value_min = validator("value_min", pre=True, allow_reuse=True)(check_int)
    _validate_value_max = validator("value_max", pre=True, allow_reuse=True)(check_int)


class ValueDuration(ValueNumeric[float, TimeUnit]):
    """
    A float value with a unit of type TimeUnit.
    """


class ValueFrequency(Value):
    """
    A non-negative integer value with no unit, with a Period.
    """

    frequency: ValueCount
    period: ValuePeriod

    def to_sql(
        self,
        table: TableClause | None,
        column_name: str = "value_as_number",
        with_unit: bool = True,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.
        """
        raise NotImplementedError()
