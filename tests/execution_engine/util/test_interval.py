import re

import pytest
from pandas import Timestamp
from portion import Bound, inf

from execution_engine.util.interval import Atomic
from execution_engine.util.interval import IntervalType
from execution_engine.util.interval import IntervalType as T
from execution_engine.util.interval import (
    IntervalWithType,
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


def parse_intervals(s):
    lines = [line.split("|")[1:-1] for line in s.split("\n") if line.strip() != ""]
    data = "\n".join([line[0] for line in lines[1:]])
    expected = "\n".join([line[1] for line in lines[1:]])

    intervals = parse_graphic_intervals(data)
    expected_intervals = parse_graphic_intervals(expected)

    return intervals, IntInterval(*expected_intervals)


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

    def test_interval_union_changed_priority_order(self):
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
        with IntervalType.custom_union_priority_order(
            IntervalType.intersection_priority()
        ):
            interval_union = interval1 | interval2

        assert len(interval_union) == 2
        assert interval_union[0] == interval_datetime(
            Timestamp("2023-02-28 23:00:00+0000", tz="UTC"),
            Timestamp("2023-03-02 16:59:59+0000", tz="UTC"),
            T.POSITIVE,
        )
        assert interval_union[1] == interval_datetime(
            Timestamp("2023-03-02 17:00:00+0000", tz="UTC"),
            Timestamp("2023-03-03 17:00:00+0000", tz="UTC"),
            T.NEGATIVE,
        )

    def test_interval_union_overlapping(self):
        def assert_intervals_equal(s):
            intervals, expected_intervals = parse_intervals(s)
            interval_union = intervals[0] | intervals[1]
            assert interval_union == expected_intervals

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

    def test_interval_union_overlapping_changed_priority_order(self):
        def assert_intervals_equal(s):
            intervals, expected_intervals = parse_intervals(s)

            with IntervalType.custom_union_priority_order(
                IntervalType.intersection_priority()
            ):
                interval_union = intervals[0] | intervals[1]

            assert interval_union == expected_intervals

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-----------] POSITIVE | [---)   (---] POSITIVE |
        |     [---]     NEGATIVE |     [---]     NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-----------] NEGATIVE | [-----------] NEGATIVE |
        |     [---]     POSITIVE |               POSITIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-------]     POSITIVE | [---)         POSITIVE |
        |     [-------] NEGATIVE |     [-------] NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2---3---4          | 1---2---3---4          |
        | [-------]     NEGATIVE | [-------]     NEGATIVE |
        |     [-------] POSITIVE |         (---] POSITIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2-------4          | 1---2-------4          |
        | [---]         POSITIVE | [---)         POSITIVE |
        |     [-------] NEGATIVE |     [-------] NEGATIVE |
        """
        assert_intervals_equal(data)

        data = """
        |  INTERVALS             | EXPECTED               |
        | 1---2-------4          | 1---2-------4          |
        | [---]         NEGATIVE | [---]        NEGATIVE  |
        |     [-------] POSITIVE |     (-------] POSITIVE |
        """
        assert_intervals_equal(data)

    def test_interval_union_edge_case(self):
        # [[Timestamp('2023-03-02 14:00:01+0000', tz='UTC'),Timestamp('2023-03-02 19:00:00+0000', tz='UTC'); NEGATIVE],
        #  [Timestamp('2023-03-02 13:00:01+0000', tz='UTC'),Timestamp('2023-03-02 14:00:00+0000', tz='UTC'); POSITIVE],
        #  [Timestamp('2023-03-02 13:00:01+0000', tz='UTC'),Timestamp('2023-03-02 19:00:00+0000', tz='UTC'); NEGATIVE]]

        interval1 = interval_datetime(
            Timestamp("2023-03-02 14:00:01+0000", tz="UTC"),
            Timestamp("2023-03-02 19:00:00+0000", tz="UTC"),
            T.NEGATIVE,
        )
        interval2 = interval_datetime(
            Timestamp("2023-03-02 13:00:01+0000", tz="UTC"),
            Timestamp("2023-03-02 14:00:00+0000", tz="UTC"),
            T.POSITIVE,
        )
        interval3 = interval_datetime(
            Timestamp("2023-03-02 13:00:01+0000", tz="UTC"),
            Timestamp("2023-03-02 19:00:00+0000", tz="UTC"),
            T.NEGATIVE,
        )

        interval_union = interval1 | interval2 | interval3
        interval_union_different_order = interval1 | interval3 | interval2

        assert interval_union == interval_union_different_order

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
    def test_repr(self):
        assert repr(IntervalType.POSITIVE) == "POSITIVE"

    def test_str(self):
        assert str(IntervalType.NEGATIVE) == "NEGATIVE"

    def test_invert(self):
        assert ~IntervalType.POSITIVE == IntervalType.NEGATIVE
        assert ~IntervalType.NEGATIVE == IntervalType.POSITIVE
        assert ~IntervalType.NO_DATA == IntervalType.NO_DATA
        assert ~IntervalType.NOT_APPLICABLE == IntervalType.NOT_APPLICABLE

        assert ~~IntervalType.POSITIVE == IntervalType.POSITIVE
        assert ~~IntervalType.NEGATIVE == IntervalType.NEGATIVE
        assert ~~IntervalType.NO_DATA == IntervalType.NO_DATA
        assert ~~IntervalType.NOT_APPLICABLE == IntervalType.NOT_APPLICABLE

    def test_union_priority(self):
        assert IntervalType.union_priority() == [
            IntervalType.POSITIVE,
            IntervalType.NO_DATA,
            IntervalType.NOT_APPLICABLE,
            IntervalType.NEGATIVE,
        ]

    def test_intersection_priority(self):
        assert IntervalType.intersection_priority() == [
            IntervalType.NEGATIVE,
            IntervalType.POSITIVE,
            IntervalType.NO_DATA,
            IntervalType.NOT_APPLICABLE,
        ]

    def test_least_intersection_priority(self):
        assert IntervalType.least_intersection_priority() == IntervalType.NOT_APPLICABLE

    def test_custom_union_priority_order(self):
        with IntervalType.custom_union_priority_order(
            [
                IntervalType.NEGATIVE,
                IntervalType.POSITIVE,
                IntervalType.NO_DATA,
                IntervalType.NOT_APPLICABLE,
            ]
        ):
            assert IntervalType.union_priority() == [
                IntervalType.NEGATIVE,
                IntervalType.POSITIVE,
                IntervalType.NO_DATA,
                IntervalType.NOT_APPLICABLE,
            ]
        assert IntervalType.union_priority() == [
            IntervalType.POSITIVE,
            IntervalType.NO_DATA,
            IntervalType.NOT_APPLICABLE,
            IntervalType.NEGATIVE,
        ]

    def test_custom_intersection_priority_order(self):
        with IntervalType.custom_intersection_priority_order(
            [
                IntervalType.POSITIVE,
                IntervalType.NEGATIVE,
                IntervalType.NO_DATA,
                IntervalType.NOT_APPLICABLE,
            ]
        ):
            assert IntervalType.intersection_priority() == [
                IntervalType.POSITIVE,
                IntervalType.NEGATIVE,
                IntervalType.NO_DATA,
                IntervalType.NOT_APPLICABLE,
            ]
        assert IntervalType.intersection_priority() == [
            IntervalType.NEGATIVE,
            IntervalType.POSITIVE,
            IntervalType.NO_DATA,
            IntervalType.NOT_APPLICABLE,
        ]

    def test_custom_union_priority_order_exception(self):
        with pytest.raises(ValueError):
            with IntervalType.custom_union_priority_order(
                [IntervalType.POSITIVE, "INVALID"]
            ):
                pass

    def test_custom_intersection_priority_order_exception(self):
        with pytest.raises(ValueError):
            with IntervalType.custom_intersection_priority_order(
                [IntervalType.POSITIVE, "INVALID"]
            ):
                pass

    def test_custom_invert_map(self):
        test_map = {
            IntervalType.POSITIVE: IntervalType.NO_DATA,
            IntervalType.NEGATIVE: IntervalType.NOT_APPLICABLE,
            IntervalType.NO_DATA: IntervalType.POSITIVE,
            IntervalType.NOT_APPLICABLE: IntervalType.NEGATIVE,
        }

        with IntervalType.custom_invert_map(test_map):
            assert ~IntervalType.POSITIVE == IntervalType.NO_DATA
            assert ~IntervalType.NEGATIVE == IntervalType.NOT_APPLICABLE
            assert ~IntervalType.NO_DATA == IntervalType.POSITIVE
            assert ~IntervalType.NOT_APPLICABLE == IntervalType.NEGATIVE

        # Test that the map is restored
        self.test_invert()

    def test_custom_invert_map_exception(self):
        with pytest.raises(ValueError):
            with IntervalType.custom_invert_map({IntervalType.POSITIVE: "INVALID"}):
                pass

        with pytest.raises(ValueError):
            with IntervalType.custom_invert_map({"INVALID": IntervalType.NEGATIVE}):
                pass

    def test_or(self):
        # POSITIVE | others
        assert IntervalType.POSITIVE | IntervalType.NEGATIVE == IntervalType.POSITIVE
        assert IntervalType.NEGATIVE | IntervalType.POSITIVE == IntervalType.POSITIVE
        assert IntervalType.POSITIVE | IntervalType.NO_DATA == IntervalType.POSITIVE
        assert IntervalType.NO_DATA | IntervalType.POSITIVE == IntervalType.POSITIVE
        assert (
            IntervalType.POSITIVE | IntervalType.NOT_APPLICABLE == IntervalType.POSITIVE
        )
        assert (
            IntervalType.NOT_APPLICABLE | IntervalType.POSITIVE == IntervalType.POSITIVE
        )

        # NO_DATA | others
        assert IntervalType.NO_DATA | IntervalType.NEGATIVE == IntervalType.NO_DATA
        assert IntervalType.NEGATIVE | IntervalType.NO_DATA == IntervalType.NO_DATA
        assert (
            IntervalType.NO_DATA | IntervalType.NOT_APPLICABLE == IntervalType.NO_DATA
        )
        assert (
            IntervalType.NOT_APPLICABLE | IntervalType.NO_DATA == IntervalType.NO_DATA
        )

        # NOT_APPLICABLE | others
        assert (
            IntervalType.NOT_APPLICABLE | IntervalType.NEGATIVE
            == IntervalType.NOT_APPLICABLE
        )
        assert (
            IntervalType.NEGATIVE | IntervalType.NOT_APPLICABLE
            == IntervalType.NOT_APPLICABLE
        )

        # same types
        assert IntervalType.POSITIVE | IntervalType.POSITIVE == IntervalType.POSITIVE
        assert IntervalType.NEGATIVE | IntervalType.NEGATIVE == IntervalType.NEGATIVE
        assert IntervalType.NO_DATA | IntervalType.NO_DATA == IntervalType.NO_DATA
        assert (
            IntervalType.NOT_APPLICABLE | IntervalType.NOT_APPLICABLE
            == IntervalType.NOT_APPLICABLE
        )

    def test_and(self):
        # NEGATIVE & others
        assert IntervalType.NEGATIVE & IntervalType.POSITIVE == IntervalType.NEGATIVE
        assert IntervalType.POSITIVE & IntervalType.NEGATIVE == IntervalType.NEGATIVE
        assert IntervalType.NEGATIVE & IntervalType.NO_DATA == IntervalType.NEGATIVE
        assert IntervalType.NO_DATA & IntervalType.NEGATIVE == IntervalType.NEGATIVE
        assert (
            IntervalType.NEGATIVE & IntervalType.NOT_APPLICABLE == IntervalType.NEGATIVE
        )
        assert (
            IntervalType.NOT_APPLICABLE & IntervalType.NEGATIVE == IntervalType.NEGATIVE
        )

        # POSITIVE & others
        assert IntervalType.POSITIVE & IntervalType.NO_DATA == IntervalType.POSITIVE
        assert IntervalType.NO_DATA & IntervalType.POSITIVE == IntervalType.POSITIVE
        assert (
            IntervalType.POSITIVE & IntervalType.NOT_APPLICABLE == IntervalType.POSITIVE
        )
        assert (
            IntervalType.NOT_APPLICABLE & IntervalType.POSITIVE == IntervalType.POSITIVE
        )

        # NO_DATA & others
        assert (
            IntervalType.NO_DATA & IntervalType.NOT_APPLICABLE == IntervalType.NO_DATA
        )
        assert (
            IntervalType.NOT_APPLICABLE & IntervalType.NO_DATA == IntervalType.NO_DATA
        )

        # same types
        assert IntervalType.POSITIVE & IntervalType.POSITIVE == IntervalType.POSITIVE
        assert IntervalType.NEGATIVE & IntervalType.NEGATIVE == IntervalType.NEGATIVE
        assert IntervalType.NO_DATA & IntervalType.NO_DATA == IntervalType.NO_DATA
        assert (
            IntervalType.NOT_APPLICABLE & IntervalType.NOT_APPLICABLE
            == IntervalType.NOT_APPLICABLE
        )

    def test_bool(self):
        assert IntervalType.POSITIVE
        assert IntervalType.NO_DATA
        assert IntervalType.NOT_APPLICABLE

        assert not IntervalType.NEGATIVE

    def test_custom_bool(self):
        with IntervalType.custom_bool_true([IntervalType.NEGATIVE]):
            assert IntervalType.NEGATIVE
            assert not IntervalType.POSITIVE
            assert not IntervalType.NO_DATA
            assert not IntervalType.NOT_APPLICABLE

        # Test that the map is restored
        self.test_bool()


def parse_interval(s, default_type: IntervalType = IntervalType.POSITIVE):
    m = re.match(r"(\[|\()([\-\d\.]+), ?([\d\-\-]+)(; (\S+))?(\]|\))", s)

    left = Bound.CLOSED if m.group(1) == "[" else Bound.OPEN
    lower = float(m.group(2))
    upper = float(m.group(3))
    if m.group(5) is not None:
        type_ = {"P": T.POSITIVE, "N": T.NEGATIVE}[m.group(5)]
    else:
        type_ = default_type
    right = Bound.CLOSED if m.group(6) == "]" else Bound.OPEN

    return Atomic(left, lower, upper, right, type_)


class TestMergeable:
    _interval_class: IntervalWithType

    @classmethod
    def mergeable(cls, a: str, b: str):
        a = parse_interval(a)
        b = parse_interval(b)

        assert cls._interval_class._mergeable(a, b) == cls._interval_class._mergeable(
            b, a
        )

        if a.type != b.type:
            aT = Atomic(a.left, a.lower, a.upper, a.right, b.type)
            bT = Atomic(b.left, b.lower, b.upper, b.right, a.type)
            assert cls._interval_class._mergeable(
                a, b
            ) == cls._interval_class._mergeable(aT, bT)
            assert cls._interval_class._mergeable(
                b, a
            ) == cls._interval_class._mergeable(bT, aT)

        return cls._interval_class._mergeable(a, b)

    def parse_interval(self, s):
        return self._interval_class.from_atomic(*parse_interval(s))


class TestAbstractDiscreteIntervalWithType(TestMergeable):
    _interval_class = IntInterval

    def test_mergeable_same_type(self):
        assert self.mergeable("[1, 2]", "[2, 3]")
        assert self.mergeable("[1, 2]", "[2, 3)")
        assert self.mergeable("[1, 2]", "(2, 3]")  # diff
        assert self.mergeable("[1, 2]", "(2, 3)")  # diff

        assert self.mergeable("[1, 2)", "[2, 3]")  # diff
        assert self.mergeable("[1, 2)", "[2, 3)")  # diff
        assert not self.mergeable("[1, 2)", "(2, 3]")
        assert not self.mergeable("[1, 2)", "(2, 3)")

        assert self.mergeable("[1, 2]", "[1, 3]")
        assert self.mergeable("[1, 2]", "(1, 3)")

        assert self.mergeable("[1, 2]", "[1, 2]")
        assert self.mergeable("[1, 2]", "(1, 2)")

        assert self.mergeable("[1, 2]", "[1, 1]")
        assert self.mergeable("[1, 2]", "(1, 1)")

        assert self.mergeable("[1, 2]", "[3, 4]")  # diff
        assert not self.mergeable("[1, 2]", "(3, 4]")
        assert not self.mergeable("[1, 2)", "[3, 4]")
        assert not self.mergeable("[1, 2)", "(3, 4]")

        assert not self.mergeable("[1, 2]", "[4, 5]")
        assert not self.mergeable("[1, 2]", "(4, 5]")

    def test_mergeable_different_type(self):
        assert self.mergeable("[1, 2; P]", "[2, 3; N]")
        assert self.mergeable("[1, 2; P]", "[2, 3; N)")
        assert not self.mergeable("[1, 2; P]", "(2, 3; N]")
        assert not self.mergeable("[1, 2; P]", "(2, 3; N)")

        assert not self.mergeable("[1, 2; P)", "[2, 3; N]")
        assert not self.mergeable("[1, 2; P)", "[2, 3; N)")
        assert not self.mergeable("[1, 2; P)", "(2, 3; N]")
        assert not self.mergeable("[1, 2; P)", "(2, 3; N)")

        assert self.mergeable("[1, 2; P]", "[1, 3; N]")
        assert self.mergeable("[1, 2; P]", "(1, 3; N)")

        assert self.mergeable("[1, 2; P]", "[1, 2; N]")
        assert self.mergeable("[1, 2; P]", "(1, 2; N)")

        assert self.mergeable("[1, 2; P]", "[1, 1; N]")
        assert self.mergeable("[1, 2; P]", "(1, 1; N)")

        assert not self.mergeable("[1, 2; P]", "[3, 4; N]")
        assert not self.mergeable("[1, 2; P]", "(3, 4; N]")
        assert not self.mergeable("[1, 2; P)", "[3, 4; N]")
        assert not self.mergeable("[1, 2; P)", "(3, 4; N]")
        assert not self.mergeable("[1, 2; P]", "[4, 5; N]")
        assert not self.mergeable("[1, 2; P]", "(4, 5; N]")

    def test_union_edge_cases(self):
        interval1 = self.parse_interval("[2, 7; N]")
        interval2 = self.parse_interval("[1, 3; P]")
        interval3 = self.parse_interval("[1, 7; N]")

        interval_union = interval1 | interval2 | interval3
        interval_union_different_order = interval1 | interval3 | interval2

        assert interval_union == interval_union_different_order


class TestIntervalWithType(TestMergeable):
    _interval_class = IntervalWithType

    def test_mergeable_same_type(self):
        assert self.mergeable("[1, 2]", "[2, 3]")
        assert self.mergeable("[1, 2]", "[2, 3)")
        assert self.mergeable("[1, 2]", "(2, 3]")
        assert self.mergeable("[1, 2]", "(2, 3)")

        assert self.mergeable("[1, 2)", "[2, 3]")
        assert self.mergeable("[1, 2)", "[2, 3)")
        assert not self.mergeable("[1, 2)", "(2, 3]")
        assert not self.mergeable("[1, 2)", "(2, 3)")

        assert self.mergeable("[1, 2]", "[1, 3]")
        assert self.mergeable("[1, 2]", "(1, 3)")

        assert self.mergeable("[1, 2]", "[1, 2]")
        assert self.mergeable("[1, 2]", "(1, 2)")

        assert self.mergeable("[1, 2]", "[1, 1]")
        assert self.mergeable("[1, 2]", "(1, 1)")

        assert not self.mergeable("[1, 2]", "[3, 4]")
        assert not self.mergeable("[1, 2]", "(3, 4]")
        assert not self.mergeable("[1, 2)", "[3, 4]")
        assert not self.mergeable("[1, 2)", "(3, 4]")

        assert not self.mergeable("[1, 2]", "[4, 5]")
        assert not self.mergeable("[1, 2]", "(4, 5]")

    def test_mergeable_different_type(self):
        assert self.mergeable("[1, 2; P]", "[2, 3; N]")
        assert self.mergeable("[1, 2; P]", "[2, 3; N)")
        assert not self.mergeable("[1, 2; P]", "(2, 3; N]")
        assert not self.mergeable("[1, 2; P]", "(2, 3; N)")

        assert not self.mergeable("[1, 2; P)", "[2, 3; N]")
        assert not self.mergeable("[1, 2; P)", "[2, 3; N)")
        assert not self.mergeable("[1, 2; P)", "(2, 3; N]")
        assert not self.mergeable("[1, 2; P)", "(2, 3; N)")

        assert self.mergeable("[1, 2; P]", "[1, 3; N]")
        assert self.mergeable("[1, 2; P]", "(1, 3; N)")

        assert self.mergeable("[1, 2; P]", "[1, 2; N]")
        assert self.mergeable("[1, 2; P]", "(1, 2; N)")

        assert self.mergeable("[1, 2; P]", "[1, 1; N]")
        assert self.mergeable("[1, 2; P]", "(1, 1; N)")

        assert not self.mergeable("[1, 2; P]", "[3, 4; N]")
        assert not self.mergeable("[1, 2; P]", "(3, 4; N]")
        assert not self.mergeable("[1, 2; P)", "[3, 4; N]")
        assert not self.mergeable("[1, 2; P)", "(3, 4; N]")
        assert not self.mergeable("[1, 2; P]", "[4, 5; N]")
        assert not self.mergeable("[1, 2; P]", "(4, 5; N]")
