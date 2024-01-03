from pydantic import validator
from pydantic.types import NonNegativeInt

from execution_engine.util.enum import TimeUnit
from execution_engine.util.value.value import ValueNumeric, check_int


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


class ValueFrequency(ValueNumeric[NonNegativeInt, None]):
    """
    A non-negative integer value with no unit.
    """

    unit: None = None

    _validate_value = validator("value", pre=True, allow_reuse=True)(check_int)
    _validate_value_min = validator("value_min", pre=True, allow_reuse=True)(check_int)
    _validate_value_max = validator("value_max", pre=True, allow_reuse=True)(check_int)
