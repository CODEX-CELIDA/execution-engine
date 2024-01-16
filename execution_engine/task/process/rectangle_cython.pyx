from collections import namedtuple

cimport numpy as np
import numpy as np

from execution_engine.task.process import Interval
from execution_engine.util.interval import IntervalType

#Interval = namedtuple("Interval", ["lower", "upper", "type"])

DEF SCHAR_MIN = -128
DEF SCHAR_MAX = 127


def intervals_to_events(
    intervals: list[Interval],
) -> list[tuple[int, bool, IntervalType]]:
    """
    Converts the intervals to a list of events.

    The events are a sorted list of the opening/closing points of all rectangles.

    :param intervals: The intervals.
    :return: The events.
    """
    events = [(i.lower, True, i.type) for i in intervals] + [
        (i.upper + 1, False, i.type) for i in intervals
    ]
    return sorted(
        events,
        key=lambda i: (i[0]),
    )


def intersect_rects(list[Interval] intervals) -> list[Interval]:
    cdef double x_min = -np.inf
    cdef signed char y_min = SCHAR_MAX
    cdef double end_point = np.inf

    if not len(intervals):
        return []

    order = IntervalType.intersection_priority()
    events = intervals_to_events(intervals)

    for cur_x, start_point, y_max_type in events:

        y_max = order.index(y_max_type)

        if start_point:
            if end_point < cur_x:
                # we already hit an endpoint and here starts a new one, so the intersection is empty
                return []

            if y_max < y_min:
                y_min = y_max

            if cur_x > x_min:
                # this point is further than the previously found one, so reset the intersection's start point
                x_min = cur_x

        else:
            # we found and endpoint
            if cur_x > end_point:
                # this endpoint lies behind another endpoint, we know we can stop
                if x_min > end_point - 1:
                    return []
                return [Interval(lower=x_min, upper=end_point - 1, type=order[y_min])]
            end_point = cur_x

    return [Interval(lower=x_min, upper=end_point - 1, type=order[y_min])]


def union_rects(list[Interval] intervals) -> list[Interval]:
    cdef double last_x = -np.inf
    cdef double last_x_closed = -np.inf
    cdef double cur_x = -np.inf
    cdef double first_x
    cdef signed char max_open_y = SCHAR_MIN
    #cdef signed char open_y[len(intervals)]

    if not intervals:
        return []

    order = IntervalType.union_priority()[::-1]

    events = intervals_to_events(intervals)
    first_x = events[0][0]

    #open_y = SortedList()
    open_y = list()

    union = []

    for x_min, start_point, y_max_type in events:
        y_max = order.index(y_max_type)

        if x_min > cur_x:
            # previously unvisited x
            cur_x = x_min

            if start_point:
                # start of a rectangle, check if this current y_max is bigger than any of the currently open and if so, start a new rectangle

                if not open_y:  # no currently open rectangles
                    last_x = cur_x  # start new output rectangle
                elif y_max > max_open_y:
                    union.append(
                        Interval(lower=last_x, upper=cur_x - 1, type=order[max_open_y])
                    )  # close the previous rectangle at the max(y) of the open rectangles
                    last_x_closed = cur_x
                    last_x = cur_x  # start new output rectangle

                open_y.append(y_max)
                max_open_y = max(open_y)
            else:
                # end of a rectangle, check if this rectangle's y_max is bigger than any of the remaining ones and if so, start a new rectangle
                open_y.remove(y_max)
                if y_max == max_open_y:
                    max_open_y = max(open_y) if open_y else SCHAR_MIN

                if (open_y and max_open_y < y_max) or not open_y:
                    union.append(
                        Interval(lower=last_x, upper=cur_x - 1, type=order[y_max])
                    )  # close the previous rectangle at y_max
                    last_x_closed = cur_x
                    last_x = cur_x  # start new output rectangle
        else:
            # previously visited x, we possibly need to update the current's y?
            if start_point:

                # check if this y_max is bigger than any of the currently open ones, if so, start a new rectangle
                # but only if
                if y_max > max_open_y and cur_x > first_x:
                    # the newly starting rectangle has a higher y_max than the currently open ones,
                    # so we need to start a new rectangle
                    union.append(
                        Interval(lower=last_x, upper=cur_x - 1, type=order[max_open_y])
                    )
                    last_x_closed = cur_x
                    last_x = cur_x  # start new output rectangle

                open_y.append(y_max)
                max_open_y = max(open_y)
            else:
                # end of a rectangle, start new output rectangle if the remaining open rectangles have a lower y_max
                open_y.remove(y_max)
                if y_max == max_open_y:
                    max_open_y = max(open_y) if open_y else SCHAR_MIN

                if (
                    (open_y and max_open_y < y_max) or not open_y
                ) and cur_x > last_x_closed:
                    union.append(
                        Interval(lower=last_x, upper=cur_x - 1, type=order[y_max])
                    )  # close the previous rectangle at y_max
                    last_x_closed = cur_x
                    last_x = cur_x  # start new output rectangle
    return union


def intersect_interval_lists(
    left: list[Interval], right: list[Interval]
) -> list[Interval]:
    """
    Intersects each interval in the left list with each interval in the right list.

    :param left: The left list.
    :param right: The right list.
    :return: The list of intersections.
    """
    return union_rects(
        [item for x in left for y in right for item in intersect_rects([x, y])]
    )


def union_interval_lists(
    left: list[Interval], right: list[Interval]
) -> list[Interval]:
    """
    Unions each interval in the left list with each interval in the right list.

    :param left: The left list.
    :param right: The right list.
    :return: The list of unions.
    """
    processed = [item for x in left for y in right for item in union_rects([x, y])]

    result = union_rects(processed)

    return result
