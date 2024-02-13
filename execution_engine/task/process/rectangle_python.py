import numpy as np
from sortedcontainers import SortedList

from execution_engine.task.process import Interval, IntervalWithCount
from execution_engine.util.interval import IntervalType


def union_rects(intervals: list[Interval]) -> list[Interval]:
    """
    Unions the intervals.
    """

    if not len(intervals):
        return []

    with IntervalType.union_order():
        events = intervals_to_events(intervals)

        union = []

        first_x = events[0][0]
        last_x = -np.inf  # holds the x_min of the currently open output rectangle
        last_x_closed = -np.inf  # x variable of the last closed interval
        cur_x = -np.inf
        open_y = SortedList()

        for x_min, start_point, y_max in events:
            if x_min > cur_x:
                # previously unvisited x
                cur_x = x_min

                if start_point:
                    # start of a rectangle, check if this current y_max is bigger than any of the currently open and if so, start a new rectangle

                    if not open_y:  # no currently open rectangles
                        last_x = cur_x  # start new output rectangle
                    elif y_max > open_y[-1]:
                        union.append(
                            Interval(lower=last_x, upper=cur_x - 1, type=open_y[-1])
                        )  # close the previous rectangle at the max(y) of the open rectangles
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle

                    open_y.add(y_max)
                else:
                    # end of a rectangle, check if this rectangle's y_max is bigger than any of the remaining ones and if so, start a new rectangle
                    open_y.remove(y_max)

                    if (open_y and open_y[-1] < y_max) or not open_y:
                        union.append(
                            Interval(lower=last_x, upper=cur_x - 1, type=y_max)
                        )  # close the previous rectangle at y_max
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
            else:
                # previously visited x, we possibly need to update the current's y?
                if start_point:
                    # check if this y_max is bigger than any of the currently open ones, if so, start a new rectangle
                    # but only if
                    if y_max > open_y[-1] and cur_x > first_x:
                        # the newly starting rectangle has a higher y_max than the currently open ones,
                        # so we need to start a new rectangle
                        union.append(
                            Interval(lower=last_x, upper=cur_x - 1, type=open_y[-1])
                        )
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle

                    open_y.add(y_max)
                else:
                    # end of a rectangle, start new output rectangle if the remaining open rectangles have a lower y_max
                    open_y.remove(y_max)

                    if (
                        (open_y and open_y[-1] < y_max) or not open_y
                    ) and cur_x > last_x_closed:
                        union.append(
                            Interval(lower=last_x, upper=cur_x - 1, type=y_max)
                        )  # close the previous rectangle at y_max
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
        return union


def union_rects_with_count(intervals: list[Interval]) -> list[IntervalWithCount]:
    """
    Unions the intervals while keeping track of the count of overlapping intervals of the same type.
    """

    if not len(intervals):
        return []

    with IntervalType.union_order():
        events = intervals_to_events(intervals)

        union = []

        first_x = events[0][0]
        last_x = -np.inf  # holds the x_min of the currently open output rectangle
        last_x_closed = -np.inf  # x variable of the last closed interval
        cur_x = -np.inf
        open_y = SortedList()

        for x_min, start_point, y_max in events:
            if x_min > cur_x:
                # previously unvisited x
                cur_x = x_min

                if start_point:
                    # start of a rectangle, check if this current y_max is bigger than any
                    # of the currently open and if so, start a new rectangle

                    if not open_y:  # no currently open rectangles
                        last_x = cur_x  # start new output rectangle
                    elif y_max > open_y[-1]:
                        union.append(
                            IntervalWithCount(
                                lower=last_x, upper=cur_x - 1, type=open_y[-1], count=1
                            )
                        )  # close the previous rectangle at the max(y) of the open rectangles
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
                    elif y_max == open_y[-1]:
                        # todo: this is the same code as the previous case, reduce redundancy
                        # other open rectangles have the same max, so we need to increment the count
                        # remember that this is a previously unvisited x
                        union.append(
                            IntervalWithCount(
                                lower=last_x,
                                upper=cur_x - 1,
                                type=open_y[-1],
                                count=open_y.count(y_max),
                            )
                        )
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle

                    open_y.add(y_max)
                else:
                    # end of a rectangle, check if this rectangle's y_max is bigger than any of the
                    # remaining ones and if so, start a new rectangle
                    open_y.remove(y_max)

                    if (open_y and open_y[-1] < y_max) or not open_y:
                        # no rectangle open anymore or the remaining open rectangles have a lower y_max
                        union.append(
                            IntervalWithCount(
                                lower=last_x, upper=cur_x - 1, type=y_max, count=1
                            )
                        )  # close the previous rectangle at y_max
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
                    elif open_y and open_y[-1] == y_max:
                        # todo: again, this is the same code as above, reduce redundancy
                        # other rectangles are open and have the same y as the currently closing one
                        # we need to close the current interval and start a new one (with decreased count)
                        union.append(
                            IntervalWithCount(
                                lower=last_x,
                                upper=cur_x - 1,
                                type=y_max,
                                count=open_y.count(y_max) + 1,
                            )
                        )
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
            else:
                # previously visited x, we possibly need to update the current's y?
                if start_point:
                    # a new interval is created but only if we haven't closed one at this point already
                    if (
                        y_max >= open_y[-1]
                        and cur_x > first_x
                        and cur_x > last_x_closed
                    ):
                        # the newly starting rectangle has a higher y_max than the currently open ones,
                        # so we need to start a new rectangle
                        # count is the number of open rectangles with the previous y_max
                        union.append(
                            IntervalWithCount(
                                lower=last_x,
                                upper=cur_x - 1,
                                type=open_y[-1],
                                count=open_y.count(open_y[-1]),
                            )
                        )
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle

                    open_y.add(y_max)
                else:
                    # end of a rectangle, start new output rectangle if the remaining open rectangles have a lower y_max
                    open_y.remove(y_max)

                    if (
                        (open_y and open_y[-1] < y_max) or not open_y
                    ) and cur_x > last_x_closed:
                        union.append(
                            IntervalWithCount(
                                lower=last_x, upper=cur_x - 1, type=y_max, count=1
                            )
                        )  # close the previous rectangle at y_max
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
                    elif (open_y and open_y[-1] == y_max) and cur_x > last_x_closed:
                        union.append(
                            IntervalWithCount(
                                lower=last_x,
                                upper=cur_x - 1,
                                type=y_max,
                                count=open_y.count(y_max) + 1,
                            )
                        )  # close the previous rectangle at y_max
                        last_x_closed = cur_x
                        last_x = cur_x  # start new output rectangle
        return union


def intersect_rects(intervals: list[Interval]) -> list[Interval]:
    """
    Intersects the intervals.
    """
    with IntervalType.intersection_order():
        events = intervals_to_events(intervals)

        x_min = -np.inf  # holds the x_min of the currently open output rectangle
        y_min = np.inf
        end_point = np.inf

        for cur_x, start_point, y_max in events:
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
                    return [Interval(lower=x_min, upper=end_point - 1, type=y_min)]
                end_point = cur_x

        return [Interval(lower=x_min, upper=end_point - 1, type=y_min)]


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


def union_interval_lists(left: list[Interval], right: list[Interval]) -> list[Interval]:
    """
    Unions each interval in the left list with each interval in the right list.

    :param left: The left list.
    :param right: The right list.
    :return: The list of unions.
    """
    processed = [item for x in left for y in right for item in union_rects([x, y])]

    result = union_rects(processed)

    return result
