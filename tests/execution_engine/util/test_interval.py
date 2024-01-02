import re

import pytest
from pandas import Timestamp
from portion import Bound, inf

from execution_engine.util.interval import IntervalType
from execution_engine.util.interval import IntervalType as T
from execution_engine.util.interval import (
    IntInterval,
    empty_interval_int,
    interval_datetime,
    interval_int,
)


def parse_graphic_intervals(graphics):
    """
    Helper function to parse a string representation of intervals into a list of Intervals.

    The idea is that the intervals are represented graphically, which aids in understanding the intervals.
    e.g.:

    1---2---3---4
    [---)   (---] POSITIVE
        [---]     NEGATIVE
    [-------]     NO_DATA
        (-------] NOT_APPLICABLE
    """
    # Split the graphics into lines and extract the first line (numbers)
    lines = [line for line in graphics.split("\n") if line.strip() != ""]
    number_line = lines[0]

    # Extract numbers and their positions
    numbers = [int(char) for char in number_line if char.isdigit()]
    number_positions = [i for i, char in enumerate(number_line) if char.isdigit()]

    intervals = []

    # Process each interval line
    for line in lines[1:]:
        if line.strip():  # Ignore empty lines
            # Split line into interval graphic and type
            m = re.match(r"([\[\(\-\s\)\]]+) ([A-Z_]+)", line)
            interval_graphic, interval_type_str = m.groups()

            i = 0
            while i < len(interval_graphic):
                if interval_graphic[i] in "([":
                    left_bound = (
                        Bound.CLOSED if interval_graphic[i] == "[" else Bound.OPEN
                    )
                    start_pos = i

                    # Find the corresponding end character
                    end_pos = min(
                        interval_graphic.find(")", start_pos),
                        interval_graphic.find("]", start_pos),
                    )
                    if end_pos == -1:  # If one of them is not found
                        end_pos = max(
                            interval_graphic.find(")", start_pos),
                            interval_graphic.find("]", start_pos),
                        )

                    # Map positions to numbers
                    lower = numbers[number_positions.index(start_pos)]
                    upper = numbers[number_positions.index(end_pos)]

                    right_bound = (
                        Bound.CLOSED if interval_graphic[end_pos] == "]" else Bound.OPEN
                    )

                    # Set the interval type
                    type_ = IntervalType[interval_type_str]

                    intervals.append(
                        IntInterval.from_atomic(
                            left_bound, lower, upper, right_bound, type_
                        )
                    )

                    i = end_pos
                i += 1

    return intervals


def test_parse_graphic_intervals():
    graphic_intervals = """
    1---2---3---4
    [---)   (---] POSITIVE
        [---]     NEGATIVE
    [-------]     NO_DATA
        (-------] NOT_APPLICABLE
    """
    intervals = parse_graphic_intervals(graphic_intervals)
    assert len(intervals) == 5

    assert intervals[0] == IntInterval.from_atomic(
        Bound.CLOSED, 1, 2, Bound.OPEN, T.POSITIVE
    )
    assert intervals[1] == IntInterval.from_atomic(
        Bound.OPEN, 3, 4, Bound.CLOSED, T.POSITIVE
    )
    assert intervals[2] == IntInterval.from_atomic(
        Bound.CLOSED, 2, 3, Bound.CLOSED, T.NEGATIVE
    )
    assert intervals[3] == IntInterval.from_atomic(
        Bound.CLOSED, 1, 3, Bound.CLOSED, T.NO_DATA
    )
    assert intervals[4] == IntInterval.from_atomic(
        Bound.OPEN, 2, 4, Bound.CLOSED, T.NOT_APPLICABLE
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
            (T.POSITIVE, T.NOT_APPLICABLE, T.POSITIVE),
            (T.NO_DATA, T.NO_DATA, T.NO_DATA),
            (T.NO_DATA, T.NOT_APPLICABLE, T.NO_DATA),
            (T.NOT_APPLICABLE, T.NOT_APPLICABLE, T.NOT_APPLICABLE),
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
            (T.POSITIVE, T.NEGATIVE, T.POSITIVE),
            (T.NO_DATA, T.NEGATIVE, T.NO_DATA),
            (T.NOT_APPLICABLE, T.NEGATIVE, T.NOT_APPLICABLE),
            (T.POSITIVE, T.NO_DATA, T.POSITIVE),
            (T.POSITIVE, T.NOT_APPLICABLE, T.POSITIVE),
            (T.NO_DATA, T.NOT_APPLICABLE, T.NO_DATA),
        ],
    )
    def test_union_interval_different_type(self, type1, type2, overlapping_type):
        # type1 is the higher priority type

        assert interval_int(1, 2, type1) | interval_int(2, 3, type2) == IntInterval(
            interval_int(1, 2, type1),
            interval_int(3, 3, type2),
        )
        assert interval_int(1, 4, type1) | interval_int(2, 3, type2) == IntInterval(
            interval_int(1, 4, type1),
        )
        assert interval_int(1, 4, type1) | interval_int(2, 5, type2) == IntInterval(
            interval_int(1, 4, type1),
            interval_int(5, 5, type2),
        )
        assert interval_int(1, 4, type1) | interval_int(5, 6, type2) == IntInterval(
            interval_int(1, 4, type1),
            interval_int(5, 6, type2),
        )

        assert interval_int(1, 3, type1) | interval_int(2, 4, type2) == IntInterval(
            interval_int(1, 3, type1), interval_int(4, 4, type2)
        )

        # inverted argument order
        assert interval_int(1, 2, type2) | interval_int(2, 3, type1) == IntInterval(
            interval_int(1, 1, type2),
            interval_int(2, 3, type1),
        )
        assert interval_int(1, 4, type2) | interval_int(2, 3, type1) == IntInterval(
            interval_int(1, 1, type2),
            interval_int(2, 3, type1),
            interval_int(4, 4, type2),
        )
        assert interval_int(1, 4, type2) | interval_int(2, 5, type1) == IntInterval(
            interval_int(1, 1, type2),
            interval_int(2, 5, type1),
        )
        assert interval_int(1, 4, type2) | interval_int(5, 6, type1) == IntInterval(
            interval_int(1, 4, type2),
            interval_int(5, 6, type1),
        )

    def test_interval_union(self):
        # [Timestamp('2023-03-02 17:00:00+0000', tz='UTC'), Timestamp('2023-03-03 17:00:00+0000', tz='UTC'); NEGATIVE]
        # [Timestamp('2023-02-28 23:00:00+0000', tz='UTC'),Timestamp('2023-03-02 22:59:59+0000', tz='UTC'); POSITIVE]

        interval1 = interval_datetime(
            Timestamp("2023-03-02 17:00:00+0000", tz="UTC"),
            Timestamp("2023-03-03 17:00:00+0000", tz="UTC"),
            T.NEGATIVE,
        )
        interval2 = interval_datetime(
            Timestamp("2023-02-28 23:00:00+0000", tz="UTC"),
            Timestamp("2023-03-02 22:59:59+0000", tz="UTC"),
            T.POSITIVE,
        )

        interval_union = interval1 | interval2

        assert len(interval_union) == 2
        assert interval_union[0] == interval_datetime(
            Timestamp("2023-02-28 23:00:00+0000", tz="UTC"),
            Timestamp("2023-03-02 22:59:59+0000", tz="UTC"),
            T.POSITIVE,
        )
        assert interval_union[1] == interval_datetime(
            Timestamp("2023-03-02 23:00:00+0000", tz="UTC"),
            Timestamp("2023-03-03 17:00:00+0000", tz="UTC"),
            T.NEGATIVE,
        )

    def test_interval_union2(self):
        def parse_intervals(s):
            lines = [
                line.split("|")[1:-1] for line in s.split("\n") if line.strip() != ""
            ]
            data = "\n".join([line[0] for line in lines[1:]])
            expected = "\n".join([line[1] for line in lines[1:]])

            intervals = parse_graphic_intervals(data)
            expected_intervals = parse_graphic_intervals(expected)

            return intervals, IntInterval(*expected_intervals)

        def assert_intervals_equal(s):
            intervals, expected_intervals = parse_intervals(s)
            interval_union = intervals[0] | intervals[1]
            assert interval_union == expected_intervals

        """
        1---2---3---4   1---2---3---4   1---2---3---4
        [-----------] P [-----------] N [---)   (---]
            [---]     N 			  P     [---]
        """

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-----------] POSITIVE | [-----------] POSITIVE |
        |     [---]     NEGATIVE | 			     NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-----------] NEGATIVE | [---)   (---] NEGATIVE |
        |     [---]     POSITIVE |     [---]     POSITIVE |
        """
        assert_intervals_equal(data)

        """
        1---2---3---4   1---2---3---4   1---2---3---4
        [-------]	  P [-------]     N [---)
            [-------] N          ---] P     [-------]
        """

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-------]     POSITIVE | [-------]     POSITIVE |
        |     [-------] NEGATIVE |         (---] NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-------]     NEGATIVE | [---)        NEGATIVE  |
        |     [-------] POSITIVE |     [-------] POSITIVE |
        """
        assert_intervals_equal(data)

        """
        1---2-------4   1---2-------4   1---2-------4
        [---]         P [---]         N [---)
            [-------] N     (-------] P     [-------]
        """

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2-------4          | 1---2-------4          |
        | [---]         POSITIVE | [---]         POSITIVE |
        |     [-------] NEGATIVE |     (-------] NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2-------4          | 1---2-------4          |
        | [---]         NEGATIVE | [---)        NEGATIVE  |
        |     [-------] POSITIVE |     [-------] POSITIVE |
        """
        assert_intervals_equal(data)

    def test_invert_interval(self):
        assert ~interval_int(1, 2, T.NEGATIVE) == interval_int(
            -inf, 0, T.NEGATIVE
        ) | interval_int(3, inf, T.NEGATIVE)
        assert ~interval_int(1, 2, T.POSITIVE) == interval_int(
            -inf, 0, T.POSITIVE
        ) | interval_int(3, inf, T.POSITIVE)
        assert ~interval_int(1, 2, T.NO_DATA) == interval_int(
            -inf, 0, T.NO_DATA
        ) | interval_int(3, inf, T.NO_DATA)
        assert ~interval_int(1, 2, T.NOT_APPLICABLE) == interval_int(
            -inf, 0, T.NOT_APPLICABLE
        ) | interval_int(3, inf, T.NOT_APPLICABLE)


class TestIntervalType:
    def test_invert_interval_type(self):
        assert ~IntervalType.POSITIVE == IntervalType.NEGATIVE
        assert ~IntervalType.NEGATIVE == IntervalType.POSITIVE
        assert ~IntervalType.NO_DATA == IntervalType.NO_DATA
        assert ~IntervalType.NOT_APPLICABLE == IntervalType.NOT_APPLICABLE
