from datetime import datetime, timedelta

import pendulum
import pytest
from pydantic import ValidationError

from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import Dosage, TimeRange, Timing
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueDuration, ValueFrequency, ValuePeriod
from tests._fixtures.concept import concept_unit_mg


class TestTimeRange:
    def test_timezone_validation(self):
        with pytest.raises(ValidationError):
            TimeRange(start=datetime.now(), end=datetime.now(), name="Test")

    def test_from_tuple(self):
        start = pendulum.now("UTC")
        end = start.add(hours=2)
        tr = TimeRange.from_tuple((start, end), name="Test")
        assert tr.start == start
        assert tr.end == end
        assert tr.name == "Test"

    def test_period(self):
        start = pendulum.now("UTC")
        end = start.add(days=1)
        tr = TimeRange(start=start, end=end)
        assert tr.period.in_days() == 1

    def test_date_range(self):
        start = pendulum.now("UTC")
        end = start.add(days=1)
        tr = TimeRange(start=start, end=end)
        assert len(tr.date_range()) == 2  # start and end date

    def test_duration(self):
        start = pendulum.now("UTC")
        end = start.add(hours=3)
        tr = TimeRange(start=start, end=end)
        assert tr.duration == timedelta(hours=3)

    def test_dict(self):
        start = pendulum.now("UTC")
        end = start.add(hours=1)
        tr = TimeRange(start=start, end=end, name="Test")
        assert tr.dict() == {"Test_start_datetime": start, "Test_end_datetime": end}


class TestDosage:
    def test_dosage_creation(self):
        dosage = Dosage(
            dose=ValueNumber(value=10, unit=concept_unit_mg),
            frequency=2,
            interval=TimeUnit.WEEK,
        )
        assert dosage.dose.value == 10
        assert dosage.dose.unit == concept_unit_mg
        assert dosage.frequency == 2
        assert dosage.interval == "wk"

    def test_enum_values(self):
        dosage = Dosage(
            dose=ValueNumber(value=10, unit=concept_unit_mg),
            frequency=2,
            interval=TimeUnit.HOUR,
        )
        assert (
            dosage.interval == "h"
        )  # Check if the enum value is used instead of the enum name
        assert not isinstance(
            dosage.interval, TimeUnit
        )  # Ensure it's not returning the enum member


class TestTiming:
    def test_convert_to_value_frequency(self):
        timing = Timing(count=5, duration=None, frequency=10, period=None)
        assert isinstance(timing.count, ValueFrequency)
        assert isinstance(timing.frequency, ValueFrequency)

    def test_convert_to_value_period(self):
        timing = Timing(
            count=5, duration=None, frequency=None, period=10 * TimeUnit.DAY
        )
        assert isinstance(timing.period, ValuePeriod)
        assert timing.period.value == 10
        assert timing.period.unit == TimeUnit.DAY

    def test_convert_to_value_duration(self):
        timing = Timing(
            count=5, duration=1.5 * TimeUnit.HOUR, frequency=None, period=None
        )
        assert isinstance(timing.duration, ValueDuration)
        assert timing.duration.value == 1.5
        assert timing.duration.unit == TimeUnit.HOUR

    def test_repr_period_one(self):
        timing = Timing(
            count=5, duration=1.5 * TimeUnit.HOUR, frequency=10, period=1 * TimeUnit.DAY
        )
        assert (
            repr(timing) == "Timing(count=5, duration=1.5 HOUR, frequency=10 per DAY)"
        )

    def test_repr_period_multiple(self):
        timing = Timing(
            count=5, duration=1.5 * TimeUnit.HOUR, frequency=10, period=2 * TimeUnit.DAY
        )
        assert (
            repr(timing) == "Timing(count=5, duration=1.5 HOUR, frequency=10 per 2 DAY)"
        )
