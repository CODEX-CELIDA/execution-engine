import datetime
import logging
from typing import Callable

import numpy as np
import pytz
from sqlalchemy import CursorResult

from execution_engine.util.interval import IntervalType, interval_datetime
from execution_engine.util.types import TimeRange

from . import Interval, IntervalWithCount

try:
    from .rectangle_cython import (  # noqa: F401 -- do not remove, should be exported by this module
        intersect_interval_lists,
        merge_adjacent_intervals,
        union_interval_lists,
        union_rects,
        union_rects_with_count,
    )
except ImportError:
    logging.info("Cython rectangle module not found, using python module")
    from .rectangle_python import (  # noqa: F401 -- do not remove, should be exported by this module
        intersect_interval_lists,
        merge_adjacent_intervals,
        union_interval_lists,
        union_rects,
        union_rects_with_count,
    )

PersonIntervals = dict[int, list[Interval]]
PersonIntervalsWithCount = dict[int, list[IntervalWithCount]]


def normalize_interval(
    interval: Interval | IntervalWithCount,
) -> Interval | IntervalWithCount:
    """
    Normalizes the interval for storage in a database.

    This function is compatible with both Interval and IntervalWithCount types.

    :param interval: The interval to normalize, can be of type Interval or IntervalWithCount.
    :return: A tuple with the normalized interval, maintaining any additional fields.
    """
    # Convert timestamps to datetime objects
    normalized_lower = datetime.datetime.fromtimestamp(interval.lower, pytz.utc)
    normalized_upper = datetime.datetime.fromtimestamp(interval.upper, pytz.utc)

    return interval._replace(lower=normalized_lower, upper=normalized_upper)


def result_to_intervals(result: CursorResult) -> PersonIntervals:
    """
    Converts the result of the interval operations to a list of intervals.
    """
    person_interval = {}

    for row in result:
        if row.interval_end < row.interval_start:
            # skip intervals with negative duration
            continue
        if row.interval_start is None:
            raise ValueError("Interval start is None")
        if row.interval_end is None:
            raise ValueError("Interval end is None")

        interval = Interval(
            row.interval_start.timestamp(),
            row.interval_end.timestamp(),
            row.interval_type,
        )

        if row.person_id not in person_interval:
            person_interval[row.person_id] = [interval]
        else:
            person_interval[row.person_id].append(interval)

    for person_id in person_interval:
        person_interval[person_id] = union_rects(person_interval[person_id])

    return person_interval


def select_type(intervals: PersonIntervals, type_: IntervalType) -> PersonIntervals:
    """
    Selects the intervals of the given type.

    :param intervals: The intervals.
    :param type_: The type.
    :return: The intervals of the given type.
    """
    return {
        key: [interval for interval in intervals[key] if interval.type == type_]
        for key in intervals
    }


def to_datetime_intervals(intervals: PersonIntervals) -> PersonIntervals:
    """
    Converts the result of the interval operations to a list of intervals.
    """
    person_interval: PersonIntervals = {}

    for person_id in intervals:
        person_interval[person_id] = [
            interval_datetime(
                datetime.datetime.fromtimestamp(interval.lower),
                datetime.datetime.fromtimestamp(interval.upper),
                interval.type,
            )
            for interval in intervals[person_id]
        ]

    return person_interval


def concat_intervals(data: list[PersonIntervals]) -> PersonIntervals:
    """
    Concatenates the intervals in the DataFrames grouped by columns.

    :param data: A list of DataFrames.
    :return: A DataFrame with the concatenated intervals.
    """
    result = {}

    for arr in data:
        if not len(arr):
            continue

        for group_keys, intervals in arr.items():
            if group_keys not in result:
                result[group_keys] = intervals
            else:
                result[group_keys] = union_rects(result[group_keys] + intervals)

    return result


def forward_fill_intervals(intervals: list[Interval]) -> list[Interval]:
    """
    Forward fills the intervals in the DataFrame.

    Each interval is extended to the next interval of a _different_ type.
    I.e. if there are multiple consecutive intervals of the same type, they are merged into one interval, and the
    last interval of these is extended until the start of the next interval (which is of a different type).

    The last interval for each person is not extended (but may be merged with the previous intervals, if they
    are of the same type).

    :param data: The intervals per person.
    :return: A DataFrame with the forward filled intervals.
    """
    all_intervals = sorted(
        intervals,
        key=lambda i: (i.lower,),
    )

    filled_intervals = []

    for i in range(len(all_intervals) - 1):
        cur_interval = all_intervals[i]
        next_interval = all_intervals[i + 1]

        if cur_interval.upper >= next_interval.lower:
            filled_intervals.append(cur_interval)
        else:
            filled_intervals.append(
                Interval(
                    cur_interval.lower,
                    next_interval.lower - 1,
                    cur_interval.type,
                )
            )

    filled_intervals.append(all_intervals[-1])

    return union_rects(filled_intervals)


def forward_fill(data: PersonIntervals) -> PersonIntervals:
    """
    Forward fills the intervals in the DataFrame.

    Each interval is extended to the next interval of a _different_ type.
    I.e. if there are multiple consecutive intervals of the same type, they are merged into one interval, and the
    last interval of these is extended until the start of the next interval (which is of a different type).

    The last interval for each person is not extended (but may be merged with the previous intervals, if they
    are of the same type).

    :param data: The intervals per person.
    :return: A DataFrame with the forward filled intervals.
    """
    # return {person_id: interval.ffill() for person_id, interval in data.items()}

    result = {}

    for person_id, intervals in data.items():
        result[person_id] = forward_fill_intervals(intervals)
    return result


def complement_intervals(
    intervals: list[Interval],
    type_: IntervalType,
) -> list[Interval]:
    """
    Complements the intervals in the DataFrame.

    The complement of an interval is the interval that is not covered by the original interval.
    The complement of an interval is always of the same type as the original interval.

    :param data: The intervals per person.
    :param type_: The type of the complement intervals.
    :return: A DataFrame with the complement intervals.
    """
    if not len(intervals):
        return [Interval(-np.inf, np.inf, type_)]

    intervals = sorted(
        intervals,
        key=lambda i: (i.lower,),
    )

    complement_intervals = []

    if intervals[0].lower > -np.inf:
        complement_intervals.append(
            Interval(
                -np.inf,
                intervals[0].lower - 1,
                type_,
            )
        )

    last_upper = intervals[0].lower - 1

    for interval in intervals:
        if interval.lower > last_upper + 1:
            complement_intervals.append(
                Interval(
                    last_upper + 1,
                    interval.lower - 1,
                    type_,
                )
            )
        last_upper = interval.upper

    if last_upper < np.inf:
        complement_intervals.append(
            Interval(
                last_upper + 1,
                np.inf,
                type_,
            )
        )

    return complement_intervals


def complementary_intervals(
    data: PersonIntervals,
    reference: PersonIntervals,
    observation_window: TimeRange,
    interval_type: IntervalType,
) -> PersonIntervals:
    """
    Insert missing intervals into a dataframe, i.e. the complement of the existing intervals w.r.t. the observation
    window. Persons that are not in the dataframe but in the base dataframe are added as well (with the full
    observation window as the interval). All added intervals have the given interval_type.

    :param data: The DataFrame with the intervals.
    :param reference: The DataFrame with the base criterion. Used to determine the patients for which intervals
        are missing.
    :param observation_window: The observation window.
    :param interval_type: The type of the complementary intervals.
    :return: A DataFrame with the inserted intervals.
    """

    interval_type_missing_persons = interval_type
    baseline_interval = Interval(
        observation_window.start.timestamp(),
        observation_window.end.timestamp(),
        interval_type_missing_persons,
    )
    observation_window_mask = Interval(
        observation_window.start.timestamp(),
        observation_window.end.timestamp(),
        IntervalType.least_intersection_priority(),
    )

    result = {}
    # invert intervals in time to get the missing intervals of existing persons

    for key in data:
        result[key] = (
            # take the least of the intersection of the observation window to retain the type of the
            #   original interval
            intersect_interval_lists(
                complement_intervals(data[key], type_=interval_type),
                [observation_window_mask],
            )
        )

    # get intervals for missing persons (in df w.r.t. base)
    missing_keys = set(reference) - set(result)

    for key in missing_keys:
        result[key] = [baseline_interval]

    return result


def invert_intervals(
    data: PersonIntervals,
    reference: PersonIntervals,
    observation_window: TimeRange,
) -> PersonIntervals:
    """
    Inverts the intervals in the DataFrame.

    The inversion is performed in time and value, i.e. first the complement set of the intervals is taken (w.r.t. the
    observation window) and then the original intervals are added back in. Then, all intervals are inverted in value.

    Also, the full observation_window is added for all patients in base that are not in df.

    Note: This means that intervals for missing persons are returned as POSITIVE, because they are considered
    NEGATIVE in the complement set and then inverted.

    :param data: The DataFrame with the intervals.
    :param reference: The DataFrame with the base criterion.
    :param observation_window: The observation window.
    :return: A DataFrame with the inserted intervals.
    """
    data_c = complementary_intervals(
        data,
        reference=reference,
        observation_window=observation_window,
        interval_type=IntervalType.NEGATIVE,  # undefined intervals are always considered negative; criteria must explicitly set NO_DATA
    )

    data = concat_intervals([data, data_c])

    return {
        person_id: [
            Interval(interval.lower, interval.upper, ~interval.type)
            for interval in intervals
        ]
        for person_id, intervals in data.items()
    }


def _process_intervals(
    data: list[PersonIntervals], operator: Callable
) -> PersonIntervals:
    """
    Processes the intervals in the lists (intersect or union)

    :param data: A list of list of intervals.
    :param operator: The operation to perform on the intervals (intersect or union).
    :return: A list with the processed intervals.
    """
    if not len(data):
        return dict()

    # assert dfs is a list of dataframes
    assert isinstance(data, list) and all(
        isinstance(arr, dict) for arr in data
    ), "data must be a list of dicts"

    result = {}

    for arr in data:
        if not len(arr):
            if operator == intersect_interval_lists:
                # if the operation is intersection, an empty dataframe means that the result is empty
                return dict()
            else:
                # if the operation is union, an empty dataframe can be ignored
                continue

        for group_keys, intervals in arr.items():
            intervals = union_rects(intervals)
            if group_keys not in result:
                result[group_keys] = intervals
            else:
                result[group_keys] = operator(result[group_keys], intervals)

    return result


def union_intervals(data: list[PersonIntervals]) -> PersonIntervals:
    """
    Unions the intervals per dict key in the list.

    :param data: A list of dict of intervals.
    :return: A dict with the unioned intervals.
    """
    return _process_intervals(data, union_interval_lists)


def interval_to_interval_with_count(interval: Interval) -> IntervalWithCount:
    """
    Converts an Interval to an IntervalWithCount.
    """
    return IntervalWithCount(interval.lower, interval.upper, interval.type, 1)


def intervals_to_intervals_with_count(
    intervals: list[Interval],
) -> list[IntervalWithCount]:
    """
    Converts a list of Intervals to a list of IntervalWithCount.
    """
    return [interval_to_interval_with_count(interval) for interval in intervals]


def count_intervals(data: list[PersonIntervals]) -> PersonIntervalsWithCount:
    """
    Counts the intervals per dict key in the list.

    :param data: A list of dict of intervals.
    :return: A dict with the unioned intervals.
    """
    if not len(data):
        return dict()

    # assert dfs is a list of dataframes
    assert isinstance(data, list) and all(
        isinstance(arr, dict) for arr in data
    ), "data must be a list of dicts"

    result = {}

    for arr in data:
        if not len(arr):
            # if the operation is union, an empty dataframe can be ignored
            continue

        for group_keys, intervals in arr.items():
            intervals_with_count = intervals_to_intervals_with_count(intervals)
            intervals_with_count = union_rects_with_count(intervals_with_count)
            if group_keys not in result:
                result[group_keys] = intervals_with_count
            else:
                result[group_keys] = union_rects_with_count(
                    result[group_keys] + intervals_with_count
                )

    return result


def filter_count_intervals(
    data: PersonIntervalsWithCount,
    min_count: int | None,
    max_count: int | None,
    type_: IntervalType,
) -> PersonIntervals:
    """
    Filters the intervals per dict key in the list by count.

    :param data: A list of dict of intervals.
    :param min_count: The minimum count of the intervals.
    :param max_count: The maximum count of the intervals.
    :param type_: The type of the intervals.
    :return: A dict with the unioned intervals.
    """

    result: PersonIntervals = {}

    if min_count is None and max_count is None:
        raise ValueError("min_count and max_count cannot both be None")
    elif min_count is not None and max_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if min_count <= interval.count <= max_count and interval.type == type_
            ]
    elif min_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if min_count <= interval.count and interval.type == type_
            ]
    elif max_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if interval.count <= max_count and interval.type == type_
            ]

    return result


def intersect_intervals(data: list[PersonIntervals]) -> PersonIntervals:
    """
    Intersects the intervals per dict key in the list.

    :param data: A list of dict of intervals.
    :return: A dict with the intersected intervals.
    """
    data = filter_dicts_by_common_keys(data)

    result = _process_intervals(data, intersect_interval_lists)

    return result


# ok
def mask_intervals(
    data: PersonIntervals,
    mask: PersonIntervals,
) -> PersonIntervals:
    """
    Masks the intervals in the dict per key.

    The intervals in data are intersected with the intervals in mask on a key-wise basis. The intervals outside the mask

    :param data: The dict with intervals that should be masked
    :param mask: A dict with intervals that should be used for masking.
    :return: A dict with the masked intervals.
    """

    person_mask = {
        person_id: [
            Interval(
                interval.lower,
                interval.upper,
                IntervalType.least_intersection_priority(),
            )
            for interval in intervals
        ]
        for person_id, intervals in mask.items()
    }

    result = {}
    for person_id in data:
        # intersect every interval in data with every interval in mask
        result[person_id] = intersect_interval_lists(
            data[person_id], person_mask[person_id]
        )

    return result


def filter_dicts_by_common_keys(
    data: list[dict],
) -> list[dict]:
    """
    Filter the dictionaries in the list by their common keys.

    :param data: A list of dicts.
    :return: A list of dicts with the common keys.
    """

    if len(data) <= 1:
        return data

    common_keys = set(data[0].keys())

    # Intersect with keys of each subsequent dictionary
    for d in data[1:]:
        common_keys &= set(d.keys())

    filtered_dicts = [{k: d[k] for k in common_keys} for d in data]

    return filtered_dicts
