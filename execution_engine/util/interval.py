import datetime
import warnings
from collections import namedtuple
from enum import StrEnum
from typing import Any, Generic, List, Protocol, TypeVar

from portion import CLOSED, Interval
from portion.const import Bound, inf


class IntervalTypeProtocol(Protocol):
    """
    The protocol of interval type used in IntervalWithType.

    This class is not expected to be used as-is, and should be subclassed
    first.
    """

    def __invert__(self) -> "IntervalTypeProtocol":
        """
        Get the complement of the interval type.
        """
        ...

    @classmethod
    def union_priority(cls) -> list["IntervalTypeProtocol"]:
        """
        Get the priority order for union.
        """
        ...

    @classmethod
    def intersection_priority(cls) -> list["IntervalTypeProtocol"]:
        """
        Get the priority order for intersection.
        """
        ...


IntervalT = TypeVar("IntervalT", bound=Any)
IntervalTypeT = TypeVar("IntervalTypeT", bound=IntervalTypeProtocol)


class IntervalType(StrEnum):
    """
    The type of interval
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NO_DATA = "NO_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    def __repr__(self) -> str:
        """
        Get the string representation of the category.
        """
        return str(self)

    def __str__(self) -> str:
        """
        Get the string representation of the interval type.
        """
        return self.name

    def __invert__(self) -> "IntervalType":
        """
        Get the complement of the interval type.
        """
        if self == self.POSITIVE:
            return self.NEGATIVE  # type: ignore # mypy doesn't like self.NEGATIVE ("Never has no attribute 'NEGATIVE'")
        if self == self.NEGATIVE:
            return self.POSITIVE  # type: ignore # mypy doesn't like self.POSITIVE ("Never has no attribute 'POSITIVE'")
        # NO_DATA and NOT_APPLICABLE are self-inverting
        return self

    @classmethod
    def union_priority(cls) -> list["IntervalType"]:  # OR |
        """
        Return the priority order for union starting with the highest priority.
        """
        return [cls.POSITIVE, cls.NO_DATA, cls.NOT_APPLICABLE, cls.NEGATIVE]

    @classmethod
    def intersection_priority(cls) -> list["IntervalType"]:  # AND &
        """
        Return the priority order for intersection starting with the highest priority.
        """
        return [cls.NEGATIVE, cls.POSITIVE, cls.NOT_APPLICABLE, cls.NO_DATA]

    @classmethod
    def least_intersection_priority(cls) -> "IntervalType":
        """
        Return the least type that can be returned by an intersection.
        """
        return cls.intersection_priority()[-1]


Atomic = namedtuple("Atomic", ["left", "lower", "upper", "right", "type"])


class IntervalWithType(Interval, Generic[IntervalT, IntervalTypeT]):
    """
    This class represents an interval.

    An interval is an (automatically simplified) union of atomic intervals.
    It can be created with IntervalWithType.from_atomic(...) or by passing Interval
    instances to __init__.
    """

    __match_args__ = ("left", "lower", "upper", "right", "type")

    def __init__(self, *intervals: Interval | Atomic):
        """
        Create a disjunction of zero, one or more intervals.

        :param intervals: zero, one or more intervals.
        """
        self._intervals = list()

        for interval in intervals:
            if isinstance(interval, Interval):
                if not interval.empty:
                    self._intervals.extend(interval._intervals)
            else:
                raise TypeError("Parameters must be Interval instances")

        if len(self._intervals) > 0:
            # Sort intervals by lower bound, closed first.
            self._intervals.sort(key=lambda i: (i.lower, i.left is Bound.OPEN))

            i = 0
            # Try to merge consecutive intervals
            while i < len(self._intervals) - 1:
                current = self._intervals[i]
                successor = self._intervals[i + 1]

                if self.__class__._mergeable(current, successor):
                    if current.lower == successor.lower:
                        lower = current.lower
                        left = (
                            current.left
                            if current.left == Bound.CLOSED
                            else successor.left
                        )

                    else:
                        lower = min(current.lower, successor.lower)
                        left = (
                            current.left if lower == current.lower else successor.left
                        )

                    if current.upper == successor.upper:
                        upper = current.upper
                        right = (
                            current.right
                            if current.right == Bound.CLOSED
                            else successor.right
                        )
                    else:
                        upper = max(current.upper, successor.upper)
                        right = (
                            current.right if upper == current.upper else successor.right
                        )

                    union = Atomic(left, lower, upper, right, current.type)
                    self._intervals.pop(i)  # pop current
                    self._intervals.pop(i)  # pop successor
                    self._intervals.insert(i, union)
                else:
                    i = i + 1

    @classmethod
    def from_atomic(
        cls,
        left: Bound,
        lower: IntervalT,
        upper: IntervalT,
        right: Bound,
        type_: IntervalTypeT,
    ) -> "IntervalWithType":
        """
        Create an Interval instance containing a single atomic interval.

        :param left: either CLOSED or OPEN.
        :param lower: value of the lower bound.
        :param upper: value of the upper bound.
        :param right: either CLOSED or OPEN.
        :param type_: the type of the interval.
        """
        left = left if lower not in [inf, -inf] else Bound.OPEN
        right = right if upper not in [inf, -inf] else Bound.OPEN

        instance = cls()
        # Check for non-emptiness (otherwise keep instance._intervals = [])
        if lower < upper or (
            lower == upper and left == Bound.CLOSED and right == Bound.CLOSED
        ):
            instance._intervals = [Atomic(left, lower, upper, right, type_)]

        return instance

    @classmethod
    def _mergeable(cls, a: Atomic, b: Atomic) -> bool:
        """
        Tester whether two atomic intervals can be merged (i.e. they overlap or
        are adjacent).

        :param a: an atomic interval.
        :param b: an atomic interval.
        :return: True if mergeable, False otherwise.
        """
        if a.type != b.type:
            return False

        if a.lower < b.lower or (a.lower == b.lower and a.left == Bound.CLOSED):
            first, second = a, b
        else:
            first, second = b, a

        if first.upper == second.lower:
            return first.right == Bound.CLOSED or second.left == Bound.CLOSED

        return first.upper > second.lower

    @property
    def type(self) -> IntervalTypeT | None:
        """
        Return the type of the interval.

        :return: the type of the interval.
        """
        if self.empty:
            return None

        if not self.homogeneous:
            raise ValueError("Cannot compute type of a non-homogeneous interval")

        return self._intervals[0].type

    @property
    def types(self) -> set[IntervalTypeT]:
        """
        Return the types of the interval.

        :return: the types of the interval.
        """
        if self.empty:
            return set()

        return set(i.type for i in self._intervals)

    @property
    def homogeneous(self) -> bool:
        """
        Return True if the interval is homogeneous, False otherwise.

        :return: True if the interval is homogeneous, False otherwise.
        """
        return len(self.types) == 1

    @property
    def enclosure(self) -> "IntervalWithType":
        """
        Return the smallest interval composed of a single atomic interval that encloses
        the current interval. Only possible if all atomic intervals have the same type.

        :return: an Interval instance.
        """
        # assert that all atomics have the same type
        if not self.homogeneous:
            raise ValueError("Cannot compute enclosure of a non-homogeneous interval")

        return super().enclosure

    def __and__(self, other: "IntervalWithType") -> "IntervalWithType":
        """
        Return the intersection of the current interval and another interval.
        """
        if not isinstance(other, Interval):
            return NotImplemented

        if self.upper < other.lower or self.lower > other.upper:
            # Early out for non-overlapping intervals
            return self.__class__()
        elif self.atomic and other.atomic:
            if self.lower == other.lower:
                lower = self.lower
                left = self.left if self.left == Bound.OPEN else other.left
            else:
                lower = max(self.lower, other.lower)
                left = self.left if lower == self.lower else other.left

            if self.upper == other.upper:
                upper = self.upper
                right = self.right if self.right == Bound.OPEN else other.right
            else:
                upper = min(self.upper, other.upper)
                right = self.right if upper == self.upper else other.right

            # Determine type based on intersection priority
            type_priority = IntervalType.intersection_priority()
            result_type = min(
                (self.type, other.type), key=lambda t: type_priority.index(t)
            )
            return self.__class__.from_atomic(left, lower, upper, right, result_type)

        else:
            intersections = []

            i_iter = iter(self)
            o_iter = iter(other)
            i_current = next(i_iter, None)
            o_current = next(o_iter, None)

            while i_current is not None and o_current is not None:
                if i_current < o_current:
                    i_current = next(i_iter, None)
                elif o_current < i_current:
                    o_current = next(o_iter, None)
                else:
                    # i_current and o_current have an overlap
                    intersections.append(i_current & o_current)

                    if i_current <= o_current:
                        # o_current can still intersect next i
                        i_current = next(i_iter, None)
                    elif o_current <= i_current:
                        # i_current can still intersect next o
                        o_current = next(o_iter, None)
                    else:
                        assert False

            return self.__class__(*intersections)

    def __or__(self, other: "IntervalWithType") -> "IntervalWithType":
        """
        Return the union of the current interval and another interval.
        """
        if not isinstance(other, Interval):
            return NotImplemented

        all_intervals = sorted(
            self._intervals + other._intervals,
            key=lambda i: (i.lower, i.left == Bound.OPEN),
        )
        merged_intervals: list[Atomic] = []

        for atomic in all_intervals:
            if not merged_intervals:
                merged_intervals.append(atomic)
                continue

            last = merged_intervals[-1]

            if self._overlaps_atomic(last, atomic):
                # Split overlapping intervals
                split_intervals = self._split_overlapping_atomics(last, atomic)
                merged_intervals.pop()  # Remove the last interval
                merged_intervals.extend(split_intervals)  # Add the split intervals
            else:
                merged_intervals.append(atomic)

        return self.__class__(
            *[self.__class__.from_atomic(*i) for i in merged_intervals]
        )

    @staticmethod
    def _overlaps_atomic(a: Atomic, b: Atomic) -> bool:
        """
        Check if two atomic intervals overlap.
        """
        return a.upper > b.lower and b.upper > a.lower

    @staticmethod
    def _split_overlapping_atomics(a: Atomic, b: Atomic) -> List[Atomic]:
        """
        Split two overlapping atomic intervals into non-overlapping intervals.
        """
        sorted_intervals = sorted([a, b], key=lambda i: i.lower)
        first, second = sorted_intervals

        split_points = sorted(
            set([first.lower, first.upper, second.lower, second.upper])
        )
        result_intervals = []

        type_priority = IntervalType.union_priority()

        for i in range(len(split_points) - 1):
            lower = split_points[i]
            upper = split_points[i + 1]

            # Determine the interval type based on the overlap
            if (
                lower >= first.lower
                and upper <= first.upper
                and lower >= second.lower
                and upper <= second.upper
            ):
                # Overlapping segment
                interval_type = min(
                    (first.type, second.type), key=lambda t: type_priority.index(t)
                )
            elif lower >= first.lower and upper <= first.upper:
                # Non-overlapping segment from first
                interval_type = first.type
            else:
                # Non-overlapping segment from second
                interval_type = second.type

            result_intervals.append(
                Atomic(Bound.CLOSED, lower, upper, Bound.CLOSED, interval_type)
            )

        return result_intervals

    def complement(self, type_: IntervalTypeT | None = None) -> "IntervalWithType":
        """
        Return the complement of the current interval.

        The type_ parameter is used to enforce the type of the complement, because if this interval is empty or
        non-homogeneous, the type of the complement cannot be inferred.

        :param type_: the (forced) type of the complementary interval.
        :return: the complement of the current interval.
        """

        if type_ is None:
            if self.empty:
                raise ValueError(
                    "Cannot compute complement of an empty interval (because type is not known). Supply a type."
                )
            if not self.homogeneous:
                raise ValueError(
                    "Cannot compute complement of a non-homogeneous interval (because type is not known). Supply a type."
                )
            type_ = self.type

        assert type_ is not None, "Type should not be None at this point"

        complements = [
            self.__class__.from_atomic(
                Bound.OPEN, -inf, self.lower, ~self.left, type_=type_
            ),
            self.__class__.from_atomic(
                ~self.right, self.upper, inf, Bound.OPEN, type_=type_
            ),
        ]

        for i, j in zip(self._intervals[:-1], self._intervals[1:]):
            complements.append(
                self.__class__.from_atomic(
                    ~i.right, i.upper, j.lower, ~j.left, type_=type_
                )
            )

        return self.__class__(*complements)

    def __contains__(self, item: "IntervalWithType" | IntervalT) -> bool:
        """
        Check if the current interval contains another interval or a value.
        """
        if isinstance(item, Interval):
            if item.empty:
                return True
            elif self.upper < item.lower or self.lower > item.upper:
                # Early out for non-overlapping intervals
                return False
            elif self.atomic:
                left = item.lower > self.lower or (
                    item.lower == self.lower
                    and (item.left == self.left or self.left == Bound.CLOSED)
                )
                right = item.upper < self.upper or (
                    item.upper == self.upper
                    and (item.right == self.right or self.right == Bound.CLOSED)
                )
                return left and right and item.type == self.type
            else:
                selfiter = iter(self)
                current = next(selfiter)

                for other in item:
                    while current < other:
                        try:
                            current = next(selfiter)
                        except StopIteration:
                            return False

                    # here current and other could have an overlap
                    if other not in current:
                        return False
                return True
        else:
            # Item is a value
            if self.upper < item or self.lower > item:
                return False

            for i in self._intervals:
                left = (item >= i.lower) if i.left == Bound.CLOSED else (item > i.lower)
                right = (
                    (item <= i.upper) if i.right == Bound.CLOSED else (item < i.upper)
                )
                if left and right:
                    return True
            return False

    def __invert__(self) -> "IntervalWithType":
        """
        Return the complement of the current interval.
        """
        if not self.homogeneous:
            raise ValueError("Cannot invert a non-homogeneous interval")

        type_ = self.type

        if type_ is None:
            raise ValueError(
                "Cannot invert an empty interval (because type is not known)"
            )

        return self.complement(type_)

    def __sub__(self, other: "IntervalWithType") -> "IntervalWithType":
        """
        Return the difference between the current interval and another interval.
        """
        if isinstance(other, Interval):
            return self & ~other
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        Check if the current interval is equal to another interval.
        """
        if isinstance(other, Interval):
            if len(other._intervals) != len(self._intervals):
                return False

            for a, b in zip(self._intervals, other._intervals):
                eq = (
                    a.left == b.left
                    and a.lower == b.lower
                    and a.upper == b.upper
                    and a.right == b.right
                    and a.type == b.type
                )
                if not eq:
                    return False
            return True
        else:
            return NotImplemented

    def __lt__(self, other: "IntervalWithType" | IntervalT) -> bool:
        """
        Check if the current interval is less than another interval.
        """
        if isinstance(other, Interval):
            if self.empty or other.empty:
                return False

            if self.right == Bound.OPEN or other.left == Bound.OPEN:
                return self.upper <= other.lower
            else:
                return self.upper < other.lower
        else:
            warnings.warn(
                "Comparing an interval and a value is deprecated. Convert value to singleton first.",
                DeprecationWarning,
            )
            return not self.empty and (
                self.upper < other or (self.right == Bound.OPEN and self.upper == other)
            )

    def __gt__(self, other: "IntervalWithType" | IntervalT) -> bool:
        """
        Check if the current interval is greater than another interval.
        """
        if isinstance(other, Interval):
            if self.empty or other.empty:
                return False

            if self.left == Bound.OPEN or other.right == Bound.OPEN:
                return self.lower >= other.upper
            else:
                return self.lower > other.upper
        else:
            warnings.warn(
                "Comparing an interval and a value is deprecated. Convert value to singleton first.",
                DeprecationWarning,
            )
            return not self.empty and (
                self.lower > other or (self.left == Bound.OPEN and self.lower == other)
            )

    def __le__(self, other: "IntervalWithType" | IntervalT) -> bool:
        """
        Check if the current interval is less than or equal to another interval.
        """
        if isinstance(other, Interval):
            if self.empty or other.empty:
                return False

            if self.right == Bound.OPEN or other.right == Bound.CLOSED:
                return self.upper <= other.upper
            else:
                return self.upper < other.upper
        else:
            warnings.warn(
                "Comparing an interval and a value is deprecated. Convert value to singleton first.",
                DeprecationWarning,
            )
            return not self.empty and self.upper <= other

    def __ge__(self, other: "IntervalWithType" | IntervalT) -> bool:
        """
        Check if the current interval is greater than or equal to another interval.
        """
        if isinstance(other, Interval):
            if self.empty or other.empty:
                return False

            if self.left == Bound.OPEN or other.left == Bound.CLOSED:
                return self.lower >= other.lower
            else:
                return self.lower > other.lower
        else:
            warnings.warn(
                "Comparing an interval and a value is deprecated. Convert value to singleton first.",
                DeprecationWarning,
            )
            return not self.empty and self.lower >= other

    def __hash__(self) -> int:
        """
        Get the hash of the interval.
        """
        return hash(tuple([self.lower, self.upper, self.type]))

    def __repr__(self) -> str:
        """
        Get the string representation of the interval.
        """
        if self.empty:
            return "()"

        string = []
        for interval in self._intervals:
            if interval.lower == interval.upper:
                string.append(
                    "[" + repr(interval.lower) + "; " + repr(interval.type) + "]"
                )
            else:
                string.append(
                    ("[" if interval.left == Bound.CLOSED else "(")
                    + repr(interval.lower)
                    + ","
                    + repr(interval.upper)
                    + "; "
                    + repr(interval.type)
                    + ("]" if interval.right == Bound.CLOSED else ")")
                )
        return " | ".join(string)


class AbstractDiscreteIntervalWithType(
    IntervalWithType[IntervalT, IntervalTypeT], Generic[IntervalT, IntervalTypeT]
):
    """
    An abstract class for discrete interval.

    This class is not expected to be used as-is, and should be subclassed
    first. The only attribute/method that should be overriden is the `_step`
    class variable. This variable defines the step between two consecutive
    values of the discrete domain (e.g., 1 for integers).
    If a meaningfull step cannot be provided (e.g., for characters), the
    _incr and _decr class methods can be overriden. They respectively return
    the next and previous value given the current one.

    This class is still experimental and backward incompatible changes may
    occur even in minor or patch updates of portion.
    """

    _step: Any

    @classmethod
    def _incr(cls, value: IntervalT) -> IntervalT:
        """
        Increment given value.

        :param value: value to increment.
        :return: incremented value.
        """
        return value + cls._step

    @classmethod
    def _decr(cls, value: IntervalT) -> IntervalT:
        """
        Decrement given value.

        :param value: value to decrement.
        :return: decremented value.
        """
        return value - cls._step

    @classmethod
    def from_atomic(
        cls,
        left: Bound,
        lower: IntervalT,
        upper: IntervalT,
        right: Bound,
        type_: IntervalTypeT,
    ) -> "AbstractDiscreteIntervalWithType":
        """
        Create an Interval instance containing a single atomic interval.

        :param left: either CLOSED or OPEN.
        :param lower: value of the lower bound.
        :param upper: value of the upper bound.
        :param right: either CLOSED or OPEN.
        :param type_: the type of the interval.
        """
        if left == Bound.OPEN and lower not in [-inf, inf]:
            left = Bound.CLOSED
            lower = cls._incr(lower)

        if right == Bound.OPEN and upper not in [-inf, inf]:
            right = Bound.CLOSED
            upper = cls._decr(upper)

        return super().from_atomic(left, lower, upper, right, type_)

    @classmethod
    def _mergeable(cls, a: Atomic, b: Atomic) -> bool:
        if a.type != b.type:
            return False

        if a.upper <= b.upper:
            first, second = a, b
        else:
            first, second = b, a

        if first.right == Bound.CLOSED and first.upper < second.lower:
            first = Atomic(
                first.left,
                first.lower,
                cls._incr(first.upper),
                Bound.OPEN,
                first.type,
            )

        return super()._mergeable(first, second)


class IntInterval(AbstractDiscreteIntervalWithType[int, IntervalType]):  # type: ignore # mypy thinks IntervalType is not a subtype of IntervalTypeProtocol, but it is # todo: can we fix this?
    """
    An integer interval.
    """

    _step = 1


class DateTimeInterval(
    AbstractDiscreteIntervalWithType[datetime.datetime, IntervalType]  # type: ignore # mypy thinks IntervalType is not a subtype of IntervalTypeProtocol, but it is # todo: can we fix this?
):
    """
    A datetime interval (second precision).
    """

    _step = datetime.timedelta(seconds=1)


def interval_int(lower: int, upper: int, type_: IntervalType) -> IntInterval:
    """
    Creates a new integer interval.

    :param lower: The lower bound.
    :param upper: The upper bound.
    :param type_: The type of the interval.
    :return: The new integer interval.
    """
    return IntInterval.from_atomic(CLOSED, lower, upper, CLOSED, type_)  # type: ignore # mypy expects "IntervalT", not sure why


def empty_interval_int() -> IntInterval:
    """
    Creates an empty integer interval.

    :return: The empty integer interval.
    """
    return IntInterval()


def interval_datetime(
    lower: datetime.datetime, upper: datetime.datetime, type_: IntervalType
) -> DateTimeInterval:
    """
    Creates a new datetime interval.

    :param lower: The lower bound.
    :param upper: The upper bound.
    :return: The new datetime interval.
    """
    return DateTimeInterval.from_atomic(CLOSED, lower, upper, CLOSED, type_)  # type: ignore # mypy expects "IntervalT", not sure why


def empty_interval_datetime() -> DateTimeInterval:
    """
    Creates an empty datetime interval.

    :return: The empty datetime interval.
    """
    return DateTimeInterval()


if __name__ == "__main__":
    interval1 = interval_int(1, 5, IntervalType.POSITIVE)
    interval2 = interval_int(3, 7, IntervalType.NEGATIVE)
    interval3 = interval_int(10, 15, IntervalType.NO_DATA)
    interval4 = interval_int(5, 10, IntervalType.NOT_APPLICABLE)

    # Non-overlapping union
    union_result = interval1 | interval3
    print("Non-overlapping Union:", union_result)

    # Non-overlapping intersection
    intersection_result = interval1 & interval3
    print("Non-overlapping Intersection:", intersection_result)

    # Overlapping union
    union_result = interval1 | interval2
    print("Overlapping Union:", union_result)

    # Overlapping intersection
    intersection_result = interval1 & interval2
    print("Overlapping Intersection:", intersection_result)

    # Adjacent intervals
    adjacent_result = interval1 | interval4
    print("Adjacent Intervals:", adjacent_result)

    # Complex union
    complex_union = interval1 | interval2 | interval3 | interval4
    print("Complex Union:", complex_union)

    # Complex intersection (expected to be empty)
    complex_intersection = interval1 & interval2 & interval3 & interval4
    print("Complex Intersection:", complex_intersection)

    # Verify priority order in union
    union_priority = interval2 | interval4  # NEGATIVE should win over NOT_APPLICABLE
    print("Union Priority:", union_priority)

    # Verify priority order in intersection
    intersection_priority = (
        interval1 & interval4
    )  # POSITIVE should win over NOT_APPLICABLE
    print("Intersection Priority:", intersection_priority)
