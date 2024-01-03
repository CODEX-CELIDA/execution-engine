from pydantic import validator
from pydantic.types import NonNegativeInt
from sqlalchemy import ColumnElement, TableClause

from execution_engine.util.enum import TimeUnit
from execution_engine.util.value.value import Value, ValueNumeric, check_int


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


class ValueDuration(ValueNumeric[float, TimeUnit]):
    """
    A float value with a unit of type TimeUnit.
    """


class ValueFrequency(Value):
    """
    A non-negative integer value with no unit, with a Period.
    """

    frequency: NonNegativeInt
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
