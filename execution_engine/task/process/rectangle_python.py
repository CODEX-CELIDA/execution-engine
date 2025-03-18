import random
import typing
from functools import cmp_to_key
from typing import Callable

import numpy as np
from sortedcontainers import SortedDict, SortedList

from execution_engine.task.process import Interval, IntervalWithCount, AnyInterval
from execution_engine.util.interval import IntervalType
from timeit import default_timer as timer

MODULE_IMPLEMENTATION = "python"

def test():
    def random_intervals():
        return [Interval(i + random.randint(0, 3), i + 3 + random.randint(0, 3), IntervalType.POSITIVE)
                for i in range(500000) ]
    i1 = random_intervals()
    i2 = random_intervals()
    def make_interval(start: int, end: int, intervals: typing.List[AnyInterval]):
        return Interval(start, end, IntervalType.POSITIVE)

    import cProfile
    profile = cProfile.Profile()
    profile.enable()
    start = timer()
    find_rectangles([i1, i2], make_interval)
    end = timer()
    print(f'{end - start} s')
    profile.disable()
    profile.print_stats(sort="cumtime")


def intervals_to_events(
    intervals: list[AnyInterval], closing_offset: int = 1
) -> list[tuple[int, bool, AnyInterval]]:
    """
    Converts the intervals to a list of events.

    The events are a sorted list of the opening/closing points of all rectangles.

    :param intervals: The intervals.
    :return: The events.
    """
    # This is actually slower
    # events = [None]*(2*len(intervals))
    # i = 0
    # is_sorted = True
    # previous_end = 0
    # for interval in intervals:
    #     lower = interval.lower
    #     if lower < previous_end:
    #         is_sorted = False
    #     previous_end = lower
    #     events[i + 0] = (lower,                           True,  interval)
    #     events[i + 1] = (interval.upper + closing_offset, False, interval)
    #     i += 2
    # if is_sorted:
    #     print('is sorted')
    #     return intervals
    # else:
    #     print('is sorted')
    #     return sorted(events, key=lambda i: i[0])
    events =   [ (i.lower,                  True,  i) for i in intervals ] \
             + [ (i.upper + closing_offset, False, i) for i in intervals ]
    return sorted(events,key=lambda i: i[0])


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

        for x_min, start_point, interval in events:
            y_max = interval.type
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


def union_rects_with_count(
    intervals: list[IntervalWithCount],
) -> list[IntervalWithCount]:
    """
    Unions the intervals while keeping track of the count of overlapping intervals of the same type.
    """

    if not len(intervals):
        return []

    with IntervalType.union_order():
        events = intervals_to_events(intervals)

        union = []

        last_x_start = -np.inf  # holds the x_min of the currently open output rectangle
        last_x_end = events[0][
            0
        ]  # x variable of the last closed interval (we start with the first x, so we
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

        for x, start_point, interval in events:
            y, count_event = interval.type, interval.count
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
                            lower=last_x_start,
                            upper=x - 1,
                            type=y_max,
                            count=open_y[y_max],
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
                            lower=last_x_start, upper=x - 1, type=y, count=count
                        )
                    )  # close the previous rectangle at y_max
                    last_x_end = x
                    last_x_start = x  # start new output rectangle

            previous_x_visited = x

        return merge_adjacent_intervals(union)


def merge_adjacent_intervals(
    intervals: list[IntervalWithCount],
) -> list[IntervalWithCount]:
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
        if (
            current.lower - 1 == last.upper
            and current.type == last.type
            and current.count == last.count
        ):
            # Merge the intervals by creating a new interval and updating the last element in the merged list
            merged_intervals[-1] = IntervalWithCount(
                last.lower, current.upper, last.type, last.count
            )
        else:
            # If not adjacent or different type/count, add the current interval to the merged list
            merged_intervals.append(current)

    return merged_intervals


def intersect_rects(intervals: list[Interval]) -> list[Interval]:
    """
    Intersects the intervals.
    """
    with IntervalType.intersection_order():
        events = intervals_to_events(intervals)

        x_min = -np.inf  # holds the x_min of the currently open output rectangle
        y_min = np.inf
        end_point = np.inf

        for cur_x, start_point, interval in events:
            y_max = interval.type
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
        [item for x in left
              for y in right
              for item in intersect_rects([x, y])]
    )


def union_interval_lists(left: list[Interval], right: list[Interval]) -> list[Interval]:
    """
    Unions each interval in the left list with each interval in the right list.

    :param left: The left list.
    :param right: The right list.
    :return: The list of unions.
    """
    return union_rects(left + right)


IntervalConstructor = Callable[[int, int, typing.List[AnyInterval]], AnyInterval]

def default_is_same_result(interval_constructor):
    def is_same_result(active_intervals1, active_intervals2):
        # When we have to decide whether to extend a result interval
        # or start a new one, we compare the state for the existing
        # result interval with the new state. The states are derived
        # from the respective lists of active intervals by calling
        # interval_constructor (with fake points in time) .
        return (interval_constructor(0, 0, active_intervals1)
                == interval_constructor(0, 0, active_intervals2))
    return is_same_result

def find_rectangles(all_intervals: list[list[AnyInterval]],
                    interval_constructor: IntervalConstructor,
                    is_same_result = None) \
        -> list[AnyInterval]:
    """For multiple parallel "tracks" of intervals, identify temporal
    intervals in which no change occurs on any "track". For each such
    interval, call interval_constructor to determine how the interval
    should be represented in the overall result. To this end,
    interval_constructor receives a list "active" intervals the
    elements of which are either None or an interval from
    all_intervals and returns either None or an interval. The returned
    None values and intervals are further processed into the overall
    return value by merging adjacent intervals without "payload"
    change.

    :param all_intervals: A list of intervals that are checked for overlap with the windows.
    :param interval_constructor: A callable that accepts a start time,
                                 an end time and a list of "active"
                                 intervals and returns None or an
                                 interval. The list of active
                                 intervals has the same length as
                                 all_intervals and each element is
                                 either None or an element from the
                                 corresponding list in all_intervals.
    :return: A list of intervals computed by interval_constructor such
             that adjacent intervals (i.e. without gaps between them)
             have different "payloads".
    """
    if is_same_result is None:
        is_same_result = default_is_same_result(interval_constructor)

    # Convert all intervals into a single list of events sorted by
    # time. Multiple events at the same point in time can be problem
    # here: If an interval open event and an interval close event on
    # the same track happen at the same time (which happens for
    # adjacent intervals on that track), we must order the close event
    # before the open event, otherwise our tracking of active
    # intervals would get confused.
    track_count = len(all_intervals)
    events = [
        (time, event, interval, j)
        for j, intervals in enumerate(all_intervals)
        for interval in intervals # intervals_to_events(intervals, closing_offset=0)
        for (time,event) in [(interval.lower, True), (interval.upper, False)]
    ]
    def compare_events(event1, event2):
        if event1[0] < event2[0]: # event1 is earlier
            return -1
        elif event2[0] < event1[0]: # event2 is earlier
            return 1
        elif (event1[3] == event2[3]  # at the same time and on same track,
             and event1[1] == False): # sort close events before open events
            return -1
        else: # at the same time, but different tracks => any order is fine
            return 1
    events.sort(key = cmp_to_key(compare_events))
    event_count = len(events)

    # The result will be a list of intervals produced by
    # interval_constructor.
    result = []
    previous_end = None
    def add_segment(start, end, original_intervals):
        nonlocal previous_end
        if previous_end == start and len(result) > 0:
            result[-1] = result[-1]._replace(upper=previous_end - 1)
        interval = interval_constructor(start, end, original_intervals)
        if interval is not None: # interval type negative is implicit
            result.append(interval)
            previous_end = end

    active_intervals = [None] * track_count
    def process_events_for_point_in_time(index, point_time):
        high_time = point_time
        any_open = False
        for i in range(index, event_count):
            time, open_, interval, track = events[i]
            # Since points in time for intervals are quantized to whole
            # seconds and intervals are closed (inclusive) for both start
            # and end points, two adjacent intervals like
            # [START_TIME1, 10:59:59] [11:00:00, END_TIME2]
            # have no gap between them and can be considered a single
            # continuous interval [START_TIME1, END_TIME2].
            if (point_time == time) or (point_time == time - 1):
                if time > high_time:
                    high_time = time
                any_open |= open_
            else:
                return i, time, active_intervals, high_time if any_open else high_time + 1
            active_intervals[track] = interval if open_ else None
        return None, None, None, None

    if not event_count == 0:
        # Step through event "clusters" with a common point in time and
        # emit result intervals with unchanged interval "payload".
        index, time = 0, events[0][0]
        interval_start_time = time
        index, time, interval_start_state, high_time = process_events_for_point_in_time(index, time)
        interval_start_state = interval_start_state.copy() if interval_start_state is not None else None
        while index:
            new_index, new_time, maybe_end_state, high_time = process_events_for_point_in_time(index, time)
            # Diagram for this program point:
            # |___potential_result_interval___||                 |
            #                                 index              new_index
            # interval_start_time             time               new_time
            # interval_start_state            maybe_end_state
            #                                  high_time
            if (maybe_end_state is None) or (not is_same_result(interval_start_state, maybe_end_state)):
                add_segment(interval_start_time, time, interval_start_state)
                interval_start_time = high_time
                interval_start_state = maybe_end_state.copy() if maybe_end_state is not None else None
            index, time = new_index, new_time
    return result
