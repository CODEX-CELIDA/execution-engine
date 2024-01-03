from datetime import date, datetime, timedelta
from typing import Any

import pendulum
from pydantic import BaseModel, root_validator, validator

from execution_engine.util.enum import TimeUnit
from execution_engine.util.interval import (
    DateTimeInterval,
    IntervalType,
    interval_datetime,
)
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueCount, ValueDuration, ValuePeriod


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
    def period(self) -> pendulum.Interval:
        """
        Get the period of the time range.
        """
        return pendulum.interval(start=self.start.date(), end=self.end.date())

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

    def interval(self, type_: IntervalType) -> DateTimeInterval:
        """
        Get the interval of the time range.

        :param type_: The type of interval to get.
        :return: The interval.
        """
        return interval_datetime(self.start, self.end, type_=type_)

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, datetime]:
        """
        Get the dictionary representation of the time range.
        """
        prefix = self.name + "_" if self.name else ""
        return {
            prefix + "start_datetime": self.start,
            prefix + "end_datetime": self.end,
        }


class Timing(BaseModel):
    """
    The timing of a criterion.

    The criterion is satisfied _once_ if it is satisfied 'frequency' times in a certain period,
    where each time should last 'duration'. The criterion is satisfied if it is satisfied 'count' times.

    The timing of a criterion is defined by the following parameters:
    - count: The number of times that the criterion should be satisfied according to the other parameters.
    - duration: The duration of each time that the criterion should be satisfied.
    - frequency: Number of repetitions of the criterion per period (= the interval in which the criterion should be
        satisfied 'frequency' times).
    """

    count: ValueCount | None
    duration: ValueDuration | None  # from duration OR boundsRange
    frequency: ValueCount | None
    interval: ValuePeriod | None

    class Config:
        """
        Pydantic configuration.
        """

        validate_assignment = True

    @validator("count", "frequency", pre=True)
    def convert_to_value_count(cls, v: Any) -> ValueCount:
        """
        Convert the count and frequency to ValueFrequency when they are integers.

        Possible because ValueFrequency has no other parameters.
        """
        if isinstance(v, int):
            return ValueCount(value=v)
        return v

    @validator("interval", pre=True)
    def convert_to_value_period(cls, v: Any) -> ValuePeriod:
        """
        Convert the count and frequency to ValueFrequency when they are integers.

        Possible under the assumption that a single value is value, not value_min or value_max.
        """
        if isinstance(v, TimeUnit):
            return ValuePeriod(value=1, unit=v)

        return v

    @root_validator  # type: ignore
    def validate_value(cls, values: dict) -> dict:
        """
        Validate interval is set when frequency is set; set frequency when interval is set.
        """

        if values.get("interval") is None:
            if values.get("frequency") is not None:
                raise ValueError("interval must be set when frequency is set.")
        else:
            if values.get("frequency") is None:
                values["frequency"] = 1

        return values

    def __repr__(self) -> str:
        """
        Get the string representation of the timing.
        """

        return f"{self.__class__.__name__}(count{self.count}, duration{self.duration}, frequency{self.frequency} per {self.interval})"


class Dosage(Timing):
    """
    A dosage consisting of a dose in addition to the Timing fields.
    """

    dose: ValueNumber | None

    class Config:
        """
        Pydantic configuration.
        """

        # todo: why is this needed? Remove it or comment why it's needed.
        use_enum_values = True
        """ Use enum values instead of names. """
