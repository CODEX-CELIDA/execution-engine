from enum import StrEnum


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
        Get the string representation of the time interval type.
        """
        return f"{self.__class__.__name__}.{self.name}"
