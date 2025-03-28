from functools import cmp_to_key
from typing import List, Tuple, cast

import numpy as np
from sortedcontainers import SortedList

from execution_engine.task.process import (
    AnyInterval,
    GeneralizedInterval,
    Interval,
    IntervalWithCount,
)
from execution_engine.task.process.rectangle import IntervalConstructor, SameResult
from execution_engine.util.interval import IntervalType

MODULE_IMPLEMENTATION = "python"

IntervalEvent = Tuple[int, bool, AnyInterval]
IntervalEventOnTrack = Tuple[int, bool, AnyInterval, int]


def intervals_to_events(
    intervals: list[Interval], closing_offset: int = 1
) -> list[IntervalEvent]:
    """
    Converts the intervals to a list of events.

    The events are a sorted list of the opening/closing points of all rectangles.

    :param intervals: The intervals.
    :return: The events.
    """
    events = [(i.lower, True, i) for i in intervals] + [
        (i.upper + closing_offset, False, i) for i in intervals
    ]
    return sorted(events, key=lambda i: i[0])


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
        [item for x in left for y in right for item in intersect_rects([x, y])]
    )


def union_interval_lists(left: list[Interval], right: list[Interval]) -> list[Interval]:
    """
    Unions each interval in the left list with each interval in the right list.

    :param left: The left list.
    :param right: The right list.
    :return: The list of unions.
    """
    return union_rects(left + right)


def default_is_same_result(interval_constructor: IntervalConstructor) -> SameResult:
    """
    Creates an 'is_same_result' function that determines whether two sets of active intervals
    produce the same resulting interval when passed to 'interval_constructor'.

    The returned function calls:
        interval_constructor(0, 0, active_intervals1)
    and
        interval_constructor(0, 0, active_intervals2)
    and checks if the results are equal. If they match, we say they represent the “same” result.

    :param interval_constructor: An interval constructor function.
    :return:
        A function 'is_same_result' that compares the results of two different sets of active
        intervals by invoking 'interval_constructor' on each and checking for equality.
    """

    def is_same_result(
        active_intervals1: List[GeneralizedInterval],
        active_intervals2: List[GeneralizedInterval],
    ) -> bool:
        """
        Compares the resulting intervals for two sets of active intervals.

        :param active_intervals1:
            A list of intervals (or None) describing the first track’s active intervals.
        :param active_intervals2:
            A list of intervals (or None) describing the second track’s active intervals.
        :return:
            True if 'interval_constructor(0, 0, ...)' produces the same interval for
            both sets, otherwise False.
        """
        # When we have to decide whether to extend a result interval
        # or start a new one, we compare the state for the existing
        # result interval with the new state. The states are derived
        # from the respective lists of active intervals by calling
        # interval_constructor (with fake points in time).
        return interval_constructor(0, 0, active_intervals1) == interval_constructor(
            0, 0, active_intervals2
        )

    return is_same_result


def find_rectangles(
    all_intervals: list[list[AnyInterval]],
    interval_constructor: IntervalConstructor,
    is_same_result: SameResult | None = None,
) -> list[AnyInterval]:
    """
    Low-level engine for interval construction.

    For multiple parallel "tracks" of intervals, identify segments of time
    in which no change occurs on any "track". For each such segment,
    call `interval_constructor(start, end, active_intervals)` to determine
    how to represent the interval in the overall result. To this end,
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

    events: list[IntervalEventOnTrack] = [
        (time, event, interval, j)
        for j, intervals in enumerate(all_intervals)
        for interval in intervals
        for (time, event) in [(interval.lower, True), (interval.upper, False)]
    ]
    event_count = len(events)

    if event_count == 0:
        return []

    def compare_events(
        event1: IntervalEventOnTrack, event2: IntervalEventOnTrack
    ) -> int:
        """
        Sorting comparator to ensure we process events in the correct order:
          - earlier time first
          - if same time and same track, close events before open events
            (so we don't incorrectly treat a consecutive interval on the same track
             as overlapping).

        Index of event1 and event2:
        - [0]: time of event
        - [1]: opening (True) or closing (False)
        - [2]: the interval to which the event belongs
        - [3]: track index
        """
        if event1[0] < event2[0]:  # event1 is earlier
            return -1
        elif event2[0] < event1[0]:  # event2 is earlier
            return 1
        elif event1[3] == event2[3]:  # at the same time and on same track,
            if (
                event1[2] == event2[2]
            ):  # same interval (we don't check for "is" because they might be different objects, but still represent the same interval)
                return (
                    -1 if (event1[1] is True) else 1
                )  # sort open events before open events
            else:  # different intervals
                return (
                    -1 if (event1[1] is False) else 1
                )  # sort close events before open events
        else:  # at the same time, but different tracks => any order is fine
            return 1

    # Sort events chronologically according to compare_events
    events.sort(key=cmp_to_key(compare_events))

    active_intervals: list[GeneralizedInterval] = [None] * track_count

    def finalize_interval(
        interval_start_time: int,
        current_time: int,
        interval_start_state: List[GeneralizedInterval],
    ) -> None:
        """
        Appends a new time slice (interval_start_time -> current_time) to 'result_intervals',
        ensuring we don't create duplicate adjacency boundaries if the previous slice ends
        exactly where the new one starts.
        """
        if len(result_intervals) > 0:
            previous_result = result_intervals[-1]
            if previous_result[1] == interval_start_time:
                # Adjust the previous slice so it doesn't overlap or duplicate
                result_intervals[-1] = (
                    previous_result[0],
                    previous_result[1] - 1,
                    previous_result[2],
                )

        # Now finalize the current slice
        result_intervals.append(
            (interval_start_time, current_time, interval_start_state)
        )

    def process_events_for_point_in_time(
        index: int, point_time: int
    ) -> Tuple[int, int, int] | None:
        """
        Consumes events that occur at `point_time` (or effectively that boundary),
        updating 'active_intervals' for whichever track is opening or closing
        intervals at that time.

        Returns: (new_index, new_time, copy_of_active_intervals, high_time)
          - new_index: the index of the first event not processed (because it's after point_time)
          - new_time: the time of that next event
          - copy_of_active_intervals: a snapshot of 'active_intervals' after processing
          - high_time: the highest time covered by these events (may be the same as point_time
                       or point_time + 1 if we consider inclusive boundaries).

        If we run out of events entirely, returns (None, None, None, None).
        """
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

            point_interval_closing = (
                any_open
                and not open_
                and interval.lower == interval.upper == point_time
            )

            if ((point_time == time) and not point_interval_closing) or (
                open_ and (point_time == time - 1)
            ):
                if time > high_time:
                    high_time = time
                any_open |= open_
            else:
                # As soon as we find an event that’s clearly beyond the cluster at point_time,
                # we break and return
                return (
                    i,
                    time,
                    high_time if any_open else high_time + 1,
                )

            # Opening => set this track’s active interval to the new interval
            # Closing => set it to None
            active_intervals[track] = interval if open_ else None

        # If we exit the loop fully, we used all events
        return None

    # Step through event "clusters" with a common point in time and
    # emit result intervals with unchanged interval "payload".
    index: int | None = 0
    time: int | None = events[index][0]  # type: ignore[index]
    interval_start_time: int = cast(int, time)
    result_intervals: list[tuple[int, int, List[GeneralizedInterval]]] = []

    if time is None:
        # No events at all
        return []

    # process the event at index 0 at the first timepoint
    res = process_events_for_point_in_time(cast(int, index), cast(int, time))

    if res is None:
        return []

    index, time, high_time = res

    interval_start_state = active_intervals.copy()

    # The main loop: step through event clusters
    while True:
        res = process_events_for_point_in_time(index, time)
        if res is None:
            # No more events => finalize the last slice and break
            finalize_interval(interval_start_time, time, interval_start_state)
            break

        new_index, new_time, high_time = res

        # Diagram for this program point:
        # |___potential_result_interval___||                 |
        #                                 index              new_index
        # interval_start_time             time               new_time
        # interval_start_state            maybe_end_state
        #                                  high_time

        # We have a region from [interval_start_time, time) or [interval_start_time, time]
        # with 'interval_start_state' as the active intervals.
        # Decide if we finalize that region or if we can merge with the next region.
        if not is_same_result(interval_start_state, active_intervals):
            # If the active intervals changed, finalize the old slice
            finalize_interval(interval_start_time, time, interval_start_state)

            # Update interval start info.
            interval_start_time = high_time
            interval_start_state = active_intervals.copy()

        index, time = new_index, new_time

    result = []

    # Finally, convert the (start, end, intervals) slices into actual Interval objects
    for start, end, intervals in result_intervals:
        interval = interval_constructor(start, end, intervals)

        if interval is not None:
            result.append(interval)

    return result
