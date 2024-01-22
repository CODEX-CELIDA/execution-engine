import pytest
from pydantic import ValidationError

from execution_engine.util.enum import TimeUnit
from execution_engine.util.value.value import ValueNumeric


class TestTimeUnit:
    def test_string_representation(self):
        assert str(TimeUnit.SECOND) == "SECOND"
        assert str(TimeUnit.MINUTE) == "MINUTE"
        assert str(TimeUnit.HOUR) == "HOUR"
        assert str(TimeUnit.DAY) == "DAY"
        assert str(TimeUnit.WEEK) == "WEEK"
        assert str(TimeUnit.MONTH) == "MONTH"

    def test_multiplication_with_integer(self):
        result = 5 * TimeUnit.HOUR
        assert isinstance(result, ValueNumeric)
        assert isinstance(result.value, int)
        assert result.value == 5
        assert result.unit == TimeUnit.HOUR

    def test_multiplication_with_float(self):
        result = 2.5 * TimeUnit.DAY
        assert isinstance(result, ValueNumeric)
        assert result.value == 2.5
        assert result.unit == TimeUnit.DAY

    def test_multiplication_with_non_number(self):
        with pytest.raises(ValidationError):
            _ = "test" * TimeUnit.WEEK
