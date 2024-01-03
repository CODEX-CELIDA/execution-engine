from datetime import date, datetime, timedelta
from typing import Any

import pendulum
from pydantic import BaseModel, PositiveInt, validator

from execution_engine.util.enum import TimeUnit
from execution_engine.util.interval import (
    DateTimeInterval,
    IntervalType,
    interval_datetime,
)
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueDuration, ValueFrequency, ValuePeriod


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


class Dosage(BaseModel):
    """
    A dosage consisting of a dose, frequency and interval.
    """

    dose: ValueNumber
    frequency: PositiveInt
    interval: TimeUnit

    class Config:
        """
        Pydantic configuration.
        """

        use_enum_values = True
        """ Use enum values instead of names. """


class Timing(BaseModel):
    """
    The timing of a criterion.

    The criterion is satisfied _once_ if it is satisfied 'frequency' times in an interval of 'period',
    where each time should last 'duration'. The criterion is satisfied if it is satisfied 'count' times.

    The timing of a criterion is defined by the following parameters:
    - count: The number of times that the criterion should be satisfied according to the other parameters.
    - duration: The duration of each time that the criterion should be satisfied.
    - frequency: Number of repetitions of the criterion per 'period'
    - period: The interval in which the criterion should be satisfied 'frequency' times.
    """

    count: ValueFrequency | None
    duration: ValueDuration | None  # from duration OR boundsRange
    frequency: ValueFrequency | None
    period: ValuePeriod | None

    @validator("count", "frequency", pre=True)
    def convert_to_value_frequency(cls, v: Any) -> ValueFrequency:
        """
        Convert the count and frequency to ValueFrequency when they are integers.

        Possible because ValueFrequency has no other parameters.
        """
        if isinstance(v, int):
            return ValueFrequency(value=v)
        return v

    def __repr__(self) -> str:
        """
        Get the string representation of the timing.
        """

        return f"{self.__class__.__name__}(count{self.count}, duration{self.duration}, frequency{self.frequency} per {self.period})"
