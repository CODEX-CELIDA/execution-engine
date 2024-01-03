from enum import StrEnum


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
