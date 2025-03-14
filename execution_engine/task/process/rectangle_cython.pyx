import copy
from functools import reduce
import datetime
from collections import namedtuple

cimport numpy as np

import numpy as np
from sortedcontainers import SortedDict

from execution_engine.task.process import Interval, IntervalWithCount, IntervalWithTypeCounts
from execution_engine.util.interval import IntervalType

DEF SCHAR_MIN = -128
DEF SCHAR_MAX = 127

MODULE_IMPLEMENTATION = "cython"

def intervals_to_events(
    intervals: list[Interval],
    closing_offset: int = 1,
) -> list[tuple[int, bool, IntervalType]]:
    """
    Converts the intervals to a list of events.

    The events are a sorted list of the opening/closing points of all rectangles.

    :param intervals: The intervals.
    :return: The events.
    """
    events = [(i.lower, True, i.type) for i in intervals] + [
        (i.upper + closing_offset, False, i.type) for i in intervals
    ]
    return sorted(
        events,
        key=lambda i: (i[0]),
    )

def intervals_with_count_to_events(
    intervals: list[IntervalWithCount],
) -> list[tuple[int, bool, IntervalType, int]]:
    """
    Converts the intervals to a list of events.

    The events are a sorted list of the opening/closing points of all rectangles.

    :param intervals: The intervals.
    :return: The events.
    """
    events = [(i.lower, True, i.type, i.count) for i in intervals] + [
        (i.upper + 1, False, i.type, i.count) for i in intervals
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

def union_rects_with_count(list[IntervalWithCount] intervals) -> list[IntervalWithCount]:
    cdef double last_x_start = -np.inf
    cdef double last_x_end;
    cdef double previous_x_visited = -np.inf
    cdef double first_x
    cdef signed char max_open_y = SCHAR_MIN

    if not intervals:
        return []

    order = IntervalType.union_priority()[::-1]

    events = intervals_with_count_to_events(intervals)

    union = []

    last_x_start = -np.inf  # holds the x_min of the currently open output rectangle
    last_x_end = events[0][0]  # x variable of the last closed interval (we start with the first x, so we
                                  # don't close the first rectangle at the first x)
    previous_x_visited = -np.inf
    open_y = SortedDict()

    def get_y_max() -> IntervalType | None:
        max_key = None
        for key in reversed(open_y):
            if open_y[key] > 0:
                max_key = key
                break
        return max_key

    for x, start_point, y_type, count_event in events:
        y = order.index(y_type)
        if start_point:
            y_max = get_y_max()

            if x > previous_x_visited and y_max is None:
                # no currently open rectangles
                last_x_start = x  # start new output rectangle
            elif y >= y_max:
                if x == last_x_end or x == last_x_start:
                    # we already closed a rectangle at this x, so we don't need to start a new one
                    open_y[y] = open_y.get(y, 0) + count_event
                    continue

                union.append(
                    IntervalWithCount(
                        lower=last_x_start, upper=x - 1, type=order[y_max], count=open_y[y_max]
                    )
                )
                last_x_end = x
                last_x_start = x

            open_y[y] = open_y.get(y, 0) + count_event

        else:
            open_y[y] = max(open_y.get(y, 0) - count_event, 0)

            y_max = get_y_max()

            if (y_max is None or (open_y and y_max <= y)) and x > last_x_end:
                if y_max is None or y_max < y:
                    # the closing rectangle has a higher y_max than the currently open ones
                    count = count_event
                else:
                    # the closing rectangle has the same y_max as the currently open ones
                    count = open_y[y] + count_event

                union.append(
                    IntervalWithCount(
                        lower=last_x_start, upper=x - 1, type=order[y], count=count
                    )
                )  # close the previous rectangle at y_max
                last_x_end = x
                last_x_start = x  # start new output rectangle

    previous_x_visited = x

    return merge_adjacent_intervals(union)

def merge_adjacent_intervals(intervals: list[IntervalWithCount]) -> list[IntervalWithCount]:
    """
    Merges adjacent intervals in a list of IntervalWithCount namedtuples if they have the same 'type' and 'count'.

    This function assumes that the input list 'intervals' contains IntervalWithCount namedtuples, each representing
    an interval with 'lower' and 'upper' bounds, a 'type' (categorical identifier), and a 'count' (numerical value).
    It merges intervals that are adjacent (i.e., the 'upper' bound of one interval is equal to the 'lower' bound of
    the next) and have the same 'type' and 'count'. The function is designed to work with non-overlapping intervals
    that may be adjacent.

    Parameters:
    - intervals (List[IntervalWithCount]): A list of IntervalWithCount namedtuples. Each namedtuple should have
      four fields: 'lower' (int), 'upper' (int), 'type' (str), and 'count' (int).

    Returns:
    - List[IntervalWithCount]: A list of merged IntervalWithCount namedtuples, where adjacent intervals with the
      same 'type' and 'count' have been merged into single intervals.

    Note:
    - The input list 'intervals' is assumed to be sorted by the 'lower' bound of each interval. If this is not the
      case, unexpected behavior may occur.
    - The function does not handle overlapping intervals; it is assumed that the input list does not contain any
      overlapping intervals.

    Example:
    >>> union = [
    ...     IntervalWithCount(1, 2, 'A', 10),
    ...     IntervalWithCount(2, 3, 'A', 10),
    ...     IntervalWithCount(3, 4, 'A', 10),
    ...     IntervalWithCount(4, 5, 'B', 5),
    ...     IntervalWithCount(6, 7, 'B', 5),
    ...     IntervalWithCount(8, 9, 'C', 20),
    ... ]
    >>> merge_adjacent_intervals(union)
    [IntervalWithCount(lower=1, upper=4, type='A', count=10),
     IntervalWithCount(lower=4, upper=5, type='B', count=5),
     IntervalWithCount(lower=6, upper=7, type='B', count=5),
     IntervalWithCount(lower=8, upper=9, type='C', count=20)]
    """
    if not intervals:
        return []

    merged_intervals = [intervals[0]]

    for current in intervals[1:]:
        # Get the last interval in the merged list
        last = merged_intervals[-1]

        # Check if the current interval is adjacent to the last interval and has the same type and count
        if current.lower - 1 == last.upper and current.type == last.type and current.count == last.count:
            # Merge the intervals by creating a new interval and updating the last element in the merged list
            merged_intervals[-1] = IntervalWithCount(last.lower, current.upper, last.type, last.count)
        else:
            # If not adjacent or different type/count, add the current interval to the merged list
            merged_intervals.append(current)

    return merged_intervals

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
    return union_rects(left + right)


def find_overlapping_windows(
    windows: list[Interval], intervals: list[Interval]
) -> list[Interval]:
    """
    Returns a list of windows that overlap with any interval in the intervals list. A window is included in the
    result if it overlaps in any part with any of the given intervals, not just where they intersect. The entire
    window is returned, not just the overlapping segment.

    :param windows: A list of windows, where each window is defined as an interval.
    :param intervals: A list of intervals that are checked for overlap with the windows.
    :return: A list of windows that have any overlap with the intervals.
    """
    # Convert all intervals and windows into events
    window_events = intervals_to_events(windows, closing_offset=0)
    interval_events = intervals_to_events(intervals, closing_offset=0)

    # Here we collect interval for the intersecting windows
    intersecting_windows = []
    def add_segment(start, end, interval_type):
        intersecting_windows.append(Interval(start, end, interval_type))

    # State and "event handler" functions for state transitions:
    # inside/not inside window, inside/not inside at least one
    # interval.
    previous_event = None
    window_state = False
    any_satisfied_in_window = False
    satisfied_interval_type = None
    def window_open(event_time, interval_type):
        nonlocal previous_event, window_state, any_satisfied_in_window
        assert not window_state
        window_state = interval_type
        if satisfied_interval_type is not None:
            any_satisfied_in_window = satisfied_interval_type
        previous_event = event_time
    def window_close(event_time):
        nonlocal previous_event, window_state, any_satisfied_in_window
        assert window_state
        if window_state == IntervalType.NOT_APPLICABLE:
            interval_type = IntervalType.NOT_APPLICABLE
            add_segment(previous_event, event_time, interval_type)
        elif any_satisfied_in_window == False:
            pass
        else:
            interval_type = any_satisfied_in_window
            add_segment(previous_event, event_time, interval_type)
        window_state = False
        any_satisfied_in_window = False
        previous_event = event_time
    def interval_satisfied(event_time, interval_type):
        nonlocal satisfied_interval_type, any_satisfied_in_window
        # If we are inside a window, remember that we saw an interval
        # of type interval_type. Do not overwrite previously seen
        # higher priority types with lower priority types
        if not (satisfied_interval_type in [IntervalType.POSITIVE, IntervalType.NEGATIVE]):
            satisfied_interval_type = interval_type
        if window_state:
            # Priorities: POSITIVE > NEGATIVE > NO_DATA or NOT_APPLICABLE > no value
            if any_satisfied_in_window == IntervalType.POSITIVE:
                pass
            elif any_satisfied_in_window == IntervalType.NEGATIVE:
                if interval_type == IntervalType.POSITIVE:
                    any_satisfied_in_window = interval_type
            else:
                any_satisfied_in_window = interval_type
    def interval_unsatisfied(event_time):
        nonlocal satisfied_interval_type
        satisfied_interval_type = None

    # Use two indices to traverse the two sorted lists of events in
    # parallel. Call event handler functions for state transitions.
    def interleaved_events():
        w_idx, i_idx = 0, 0
        while True:
            window_event = window_events[w_idx] if w_idx < len(window_events) else None
            interval_event = interval_events[i_idx] if i_idx < len(interval_events) else None
            # When tied in terms of event time, use the following
            # priority for reporting events:
            # window open > interval open > interval close > window close
            if window_event and (not interval_event or window_event[0] < interval_event[0] or (
                    window_event[0] == interval_event[0] and window_event[1])):
                w_idx += 1
                yield window_event, None
            elif interval_event:
                i_idx += 1
                yield None, interval_event
            else:
                break
    active_intervals = 0
    for window_event, interval_event in interleaved_events():
        if window_event:
            time, open_, interval_type = window_event
            window_open(time, interval_type) if open_ else window_close(time)
        else:
            time, open_, type_ = interval_event
            active_intervals += (1 if open_ else -1)
            if active_intervals == 0: # 1 -> 0 transition
                interval_unsatisfied(time)
            elif active_intervals == 1: # 0 -> 1 transition
                interval_satisfied(time, type_)

    # Return the list of unique intersecting windows
    return intersecting_windows

def find_rectangles_with_count(all_intervals: list[list[Interval]]) -> list[IntervalWithTypeCounts]:
    """
    For multiple parallel "tracks" of intervals, identify temporal
    intervals in which no change occurs on any "track". For each such
    interval, report the number of active intervals grouped by
    interval type across all "tracks". When there is no interval on a
    track for a given temporal interval, act as if a negative interval
    was present there.

    :param all_intervals: A list of intervals that are checked for overlap with the windows.
    :return: A list of windows that have any overlap with the intervals.
    """
    # Convert all intervals into a list of events sorted by
    # time. Multiple events at the same point in time are not a
    # problem here: since we simply count the number of "active"
    # intervals the result does not depend on the order in which we
    # process the events.
    track_count = len(all_intervals)
    events = reduce(lambda acc, intervals: acc + intervals_to_events(intervals, closing_offset=0),
                    all_intervals,
                    [])
    events.sort(key=lambda i: i[0])

    # The result will be a list of intervals
    result = []
    def add_segment(start, end, type_counts):
        # We consider the period between 23:59:59 of a day and
        # 00:00:00 of the following day to be empty.
        if not (start == end - 1 and datetime.datetime.fromtimestamp(end).time() == datetime.time(0, 0, 0)):
            # Assume implicit negative intervals: increase the count
            # for the negative type as needed so that the overall
            # count is equal to track_count.
            missing = track_count - sum(type_counts.values())
            if missing > 0:
                type_counts[IntervalType.NEGATIVE] = type_counts.get(IntervalType.NEGATIVE, 0) + missing
            result.append(IntervalWithTypeCounts(start, end, type_counts))

    # Step through events and emit result intervals whenever the
    # counts change.
    counts = dict()
    previous_time = events[0][0]
    for (time, open_, interval_type) in events:
        if previous_time is None:
            previous_time = time
        elif not previous_time == time:
            add_segment(previous_time, time, copy.copy(counts))
            previous_time = time

        old_count = counts.get(interval_type, 0)
        counts[interval_type] = old_count + (1 if open_ else -1)

    return result
