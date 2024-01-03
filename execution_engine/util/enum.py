from enum import StrEnum
from typing import Any

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

    def __str__(self) -> str:
        """
        Returns the string representation of the TimeUnit.
        """
        return self.name

    def __rmul__(self, other: Any) -> ValueNumeric:
        """
        Multiply a number by a TimeUnit.

        :param other: The number to multiply.
        :return: A ValueNumeric object.
        """
        if isinstance(other, float) and other.is_integer() or isinstance(other, int):
            return ValueNumeric[int, TimeUnit](value=other, unit=self)
        else:
            return ValueNumeric[float, TimeUnit](value=other, unit=self)
