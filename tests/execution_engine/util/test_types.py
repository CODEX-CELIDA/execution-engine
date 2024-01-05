import json
from datetime import datetime, timedelta

import pendulum
import pytest
from pydantic import ValidationError

from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import Dosage, TimeRange, Timing
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueDuration, ValuePeriod
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
            count=5,
            duration=None,
            frequency=10,
            interval=TimeUnit.DAY,
        )
        assert dosage.dose.value == 10
        assert dosage.dose.unit == concept_unit_mg
        assert dosage.count.value == 5
        assert dosage.frequency.value == 10

    @pytest.mark.skip(
        reason="Timing in Dosage is another type that does not use the use use_enum_values flag"
    )
    def test_enum_values(self):
        dosage = Dosage(
            dose=ValueNumber(value=10, unit=concept_unit_mg),
            count=5,
            duration=None,
            frequency=10,
            interval=1 * TimeUnit.HOUR,
        )
        assert (
            dosage.interval.unit == "h"
        )  # Check if the enum value is used instead of the enum name
        assert not isinstance(
            dosage.interval.unit, TimeUnit
        )  # Ensure it's not returning the enum member

    def test_serialization(self):
        dosage = Dosage(
            dose=ValueNumber(value=10, unit=concept_unit_mg),
            count=5,
            duration=None,
            frequency=10,
            interval=1 * TimeUnit.HOUR,
        )

        json_str = json.dumps(dosage.dict())
        dict_from_str = json.loads(json_str)

        assert Dosage(**dict_from_str) == dosage


class TestTiming:
    def test_interval_must_be_set_for_frequency(self):
        with pytest.raises(ValidationError):
            Timing(count=5, frequency=10, interval=None)

    def test_frequency_is_set_for_interval(self):
        timing = Timing(count=5, frequency=None, interval=TimeUnit.DAY)
        assert timing.frequency == 1

    def test_convert_to_value_period(self):
        timing = Timing(
            count=5, duration=None, frequency=None, interval=10 * TimeUnit.DAY
        )
        assert isinstance(timing.interval, ValuePeriod)
        assert timing.interval.value == 10
        assert timing.interval.unit == TimeUnit.DAY

        timing = Timing(count=5, duration=None, frequency=None, interval=TimeUnit.DAY)
        assert isinstance(timing.interval, ValuePeriod)
        assert timing.interval.value == 1
        assert timing.interval.unit == TimeUnit.DAY

    def test_convert_to_value_duration(self):
        timing = Timing(
            count=5, duration=1.5 * TimeUnit.HOUR, frequency=None, interval=None
        )
        assert isinstance(timing.duration, ValueDuration)
        assert timing.duration.value == 1.5
        assert timing.duration.unit == TimeUnit.HOUR

    def test_str_period_one(self):
        timing = Timing(
            count=5,
            duration=1.5 * TimeUnit.HOUR,
            frequency=10,
            interval=1 * TimeUnit.DAY,
        )
        assert str(timing) == "Timing(count=5, duration=1.5 HOUR, frequency=10 per DAY)"

    def test_str_period_multiple(self):
        timing = Timing(
            count=5,
            duration=1.5 * TimeUnit.HOUR,
            frequency=10,
            interval=2 * TimeUnit.DAY,
        )
        assert (
            str(timing) == "Timing(count=5, duration=1.5 HOUR, frequency=10 per 2 DAY)"
        )

    def test_set_values_after_creation(self):
        timing = Timing(count=5)

        timing.interval = 1 * TimeUnit.DAY
        assert timing.interval == ValuePeriod(value=1, unit=TimeUnit.DAY)
        assert timing.frequency == 1  # Frequency is set to 1 when interval is set

        timing.frequency = 10
        assert timing.frequency.value == 10

        # make sure timing.frequency is not overwritten when timing.interval is set before
        timing = Timing(count=5)

        # we can't assign a frequency without an interval
        with pytest.raises(ValidationError):
            timing.frequency = 10

    def test_serialization(self):
        timing = Timing(
            count=">=5",
            frequency=10,
            interval=2 * TimeUnit.DAY,
            duration=1.5 * TimeUnit.HOUR,
        )

        json_str = json.dumps(timing.dict())
        dict_from_str = json.loads(json_str)

        assert Timing(**dict_from_str) == timing
