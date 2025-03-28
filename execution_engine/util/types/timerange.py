from datetime import date, datetime, timedelta
from typing import Any

import pendulum
import pytz
from pydantic import BaseModel, field_validator

from execution_engine.util import serializable
from execution_engine.util.interval import (
    DateTimeInterval,
    IntervalType,
    interval_datetime,
)


@serializable.register_class
class TimeRange(BaseModel):
    """
    A time range.
    """

    start: datetime
    end: datetime
    name: str | None = None

    @field_validator("start", "end")
    def check_timezone(cls, v: datetime) -> datetime:
        """
        Check that the start, end parameters are timezone-aware.
        """
        if not v.tzinfo:
            raise ValueError("Datetime object must be timezone-aware")

        # workaround to fix pd.testing.assert_frame_equal errors when tzinfo is of type
        # pydantic_core._pydantic_core.TzInfo, which somehow triggers an error when comparing a dataframe with a that
        # tz in a datetime column vs a pytz.UTC tz column.
        if v.tzinfo == pytz.UTC:
            return v.astimezone(pytz.utc)

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

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, datetime]:
        """
        Get the dictionary representation of the time range.
        """
        prefix = self.name + "_" if self.name else ""
        return {
            prefix + "start_datetime": self.start,
            prefix + "end_datetime": self.end,
        }
