from pydantic.functional_validators import field_validator
from pydantic.types import NonNegativeInt
from sqlalchemy import ColumnClause, ColumnElement
from sqlalchemy import Interval as SQLInterval
from sqlalchemy import TableClause
from sqlalchemy.sql.functions import concat, func

from execution_engine.util.enum import TimeUnit
from execution_engine.util.value import ucum_to_postgres
from execution_engine.util.value.value import (
    Value,
    ValueNumeric,
    check_int,
    check_unit_none,
    check_value_min_max_none,
)


class ValuePeriod(ValueNumeric[NonNegativeInt, TimeUnit]):
    """
    A non-negative integer value with a unit of type TimeUnit, where value_min and value_max are not allowed.
    """

    value_min: None = None
    value_max: None = None

    _validate_value = field_validator("value", mode="before")(check_int)
    _validate_no_value_min = field_validator("value_min", mode="before")(
        check_value_min_max_none
    )
    _validate_no_value_max = field_validator("value_max", mode="before")(
        check_value_min_max_none
    )

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
            concat(self.value, ucum_to_postgres[str(self.unit.value)]), SQLInterval
        )


class ValueCount(ValueNumeric[NonNegativeInt, None]):
    """
    A non-negative integer value without a unit.
    """

    unit: None = None

    _validate_value = field_validator("value", mode="before")(check_int)
    _validate_value_min = field_validator("value_min", mode="before")(check_int)
    _validate_value_max = field_validator("value_max", mode="before")(check_int)
    _validate_no_unit = field_validator("unit", mode="before")(check_unit_none)

    def supports_units(self) -> bool:
        """
        Returns true if this type of value supports units.
        """
        return False


class ValueDuration(ValueNumeric[float, TimeUnit]):
    """
    A float value with a unit of type TimeUnit.
    """

    def to_sql(
        self,
        table: TableClause | None = None,
        column_name: ColumnClause | str = "value_as_number",
        with_unit: bool = False,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.

        As ValueDuration has a TimeUnit that specifies the unit of the duration, we need to extract the EPOCH (seconds)
        from the column in the database and divide it by the TimeUnit's length in seconds to get the duration in
        the correct unit. E.g. if TimeUnit is HOUR, we need to divide the database's duration by 3600 to get the
        duration in hours.

        :param table: The table to get the column from.
        :param column_name: The name of the column to get.
        :param with_unit: Whether to include the unit in the SQL representation.
        :return: The sqlalchemy clause.
        """
        # we need to divide the duration column by the interval length in seconds
        # to be able to compare it to the defined value

        c = func.extract("EPOCH", self._get_column(table, column_name))
        c_duration_seconds = self.unit.to_sql_interval_length_seconds()

        return super().to_sql(table, c / c_duration_seconds, with_unit)


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
