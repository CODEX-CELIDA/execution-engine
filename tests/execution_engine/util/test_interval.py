import pytest

from execution_engine.util.intervalt import IntervalType as T
from execution_engine.util.intervalt import (
    IntInterval,
    empty_interval_int,
    interval_int,
)


class TestInterval:
    @pytest.mark.parametrize(
        "type1,type2,type_expected",
        [
            (T.NEGATIVE, T.NEGATIVE, T.NEGATIVE),
            (T.NEGATIVE, T.POSITIVE, T.NEGATIVE),
            (T.NEGATIVE, T.NO_DATA, T.NEGATIVE),
            (T.NEGATIVE, T.NOT_APPLICABLE, T.NEGATIVE),
            (T.POSITIVE, T.POSITIVE, T.POSITIVE),
            (T.POSITIVE, T.NO_DATA, T.POSITIVE),
            (T.POSITIVE, T.NOT_APPLICABLE, T.NOT_APPLICABLE),
            (T.NOT_APPLICABLE, T.NOT_APPLICABLE, T.NOT_APPLICABLE),
            (T.NOT_APPLICABLE, T.NO_DATA, T.NOT_APPLICABLE),
            (T.NO_DATA, T.NO_DATA, T.NO_DATA),
        ],
    )
    def test_intersect_interval_different_type(self, type1, type2, type_expected):
        assert interval_int(1, 2, type1) & interval_int(2, 3, type2) == interval_int(
            2, 2, type_expected
        )
        assert interval_int(1, 4, type1) & interval_int(2, 3, type2) == interval_int(
            2, 3, type_expected
        )
        assert interval_int(1, 4, type1) & interval_int(2, 5, type2) == interval_int(
            2, 4, type_expected
        )
        assert (
            interval_int(1, 4, type1) & interval_int(5, 6, type2)
            == empty_interval_int()
        )

        # inverted argument order
        assert interval_int(1, 2, type2) & interval_int(2, 3, type1) == interval_int(
            2, 2, type_expected
        )
        assert interval_int(1, 4, type2) & interval_int(2, 3, type1) == interval_int(
            2, 3, type_expected
        )
        assert interval_int(1, 4, type2) & interval_int(2, 5, type1) == interval_int(
            2, 4, type_expected
        )
        assert (
            interval_int(1, 4, type2) & interval_int(5, 6, type1)
            == empty_interval_int()
        )

    @pytest.mark.parametrize(
        "type_", [T.NEGATIVE, T.POSITIVE, T.NOT_APPLICABLE, T.NO_DATA]
    )
    def test_union_interval_same_type(self, type_):
        assert interval_int(1, 2, type_) | interval_int(2, 3, type_) == interval_int(
            1, 3, type_
        )
        assert interval_int(1, 4, type_) | interval_int(2, 3, type_) == interval_int(
            1, 4, type_
        )
        assert interval_int(1, 4, type_) | interval_int(2, 5, type_) == interval_int(
            1, 5, type_
        )
        assert interval_int(1, 4, type_) | interval_int(5, 6, type_) == interval_int(
            1, 6, type_
        )
        assert interval_int(4, 6, type_) | interval_int(2, 6, type_) == interval_int(
            2, 6, type_
        )

    @pytest.mark.parametrize(
        "type1,type2,overlapping_type",
        [
            (T.NEGATIVE, T.POSITIVE, T.POSITIVE),
            (T.NEGATIVE, T.NO_DATA, T.NO_DATA),
            (T.NEGATIVE, T.NOT_APPLICABLE, T.NOT_APPLICABLE),
            (T.POSITIVE, T.NO_DATA, T.POSITIVE),
            (T.POSITIVE, T.NOT_APPLICABLE, T.POSITIVE),
            (T.NOT_APPLICABLE, T.NO_DATA, T.NO_DATA),
        ],
    )
    def test_union_interval_different_type(self, type1, type2, overlapping_type):
        assert interval_int(1, 2, type1) | interval_int(2, 3, type2) == IntInterval(
            interval_int(1, 2, type1),
            interval_int(2, 3, type2),
        )
        assert interval_int(1, 4, type1) | interval_int(2, 3, type2) == IntInterval(
            interval_int(1, 2, type1),
            interval_int(2, 3, overlapping_type),
            interval_int(3, 4, type1),
        )
        assert interval_int(1, 4, type1) | interval_int(2, 5, type2) == IntInterval(
            interval_int(1, 2, type1),
            interval_int(2, 4, overlapping_type),
            interval_int(4, 5, type2),
        )
        assert interval_int(1, 4, type1) | interval_int(5, 6, type2) == IntInterval(
            interval_int(1, 4, type1),
            interval_int(5, 6, type2),
        )

        # inverted argument order
        assert interval_int(1, 2, type2) | interval_int(2, 3, type1) == IntInterval(
            interval_int(1, 2, type2),
            interval_int(2, 3, type1),
        )
        assert interval_int(1, 4, type2) | interval_int(2, 3, type1) == IntInterval(
            interval_int(1, 2, type2),
            interval_int(2, 3, overlapping_type),
            interval_int(3, 4, type2),
        )
        assert interval_int(1, 4, type2) | interval_int(2, 5, type1) == IntInterval(
            interval_int(1, 2, type2),
            interval_int(2, 4, overlapping_type),
            interval_int(4, 5, type1),
        )
        assert interval_int(1, 4, type2) | interval_int(5, 6, type1) == IntInterval(
            interval_int(1, 4, type2),
            interval_int(5, 6, type1),
        )
