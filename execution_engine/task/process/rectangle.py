import datetime
import importlib
import logging
import os
from typing import Callable, cast

import numpy as np
import pendulum
import pytz
from sqlalchemy import CursorResult

from execution_engine.util.interval import IntervalType, interval_datetime
from execution_engine.util.types import TimeRange

from . import Interval, IntervalWithCount

PROCESS_RECTANGLE_VERSION = os.getenv("PROCESS_RECTANGLE_VERSION", "auto")


if "PROCESS_RECTANGLE_VERSION" not in globals() or PROCESS_RECTANGLE_VERSION == "auto":
    try:
        # Try to import the Cython version
        importlib.import_module(
            ".rectangle_cython", package="execution_engine.task.process"
        )
        module_name = ".rectangle_cython"
    except ImportError:
        logging.info("Cython rectangle module not found, using python module")
        module_name = ".rectangle_python"
elif PROCESS_RECTANGLE_VERSION == "cython":
    module_name = ".rectangle_cython"
elif PROCESS_RECTANGLE_VERSION == "python":
    module_name = ".rectangle_python"
else:
    module_name = ".rectangle_python"  # Default to Python version

# Dynamically import the chosen module
_impl = importlib.import_module(module_name, package="execution_engine.task.process")


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
        person_interval[person_id] = _impl.union_rects(person_interval[person_id])

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
                result[group_keys] = _impl.union_rects(result[group_keys] + intervals)

    return result


def forward_fill_intervals(intervals: list[Interval]) -> list[Interval]:
    """
    Forward fills the intervals in the DataFrame.

    Each interval is extended to the next interval of a _different_ type.
    I.e. if there are multiple consecutive intervals of the same type, they are merged into one interval, and the
    last interval of these is extended until the start of the next interval (which is of a different type).

    The last interval for each person is not extended (but may be merged with the previous intervals, if they
    are of the same type).

    :param intervals: The intervals per person.
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

    return _impl.union_rects(filled_intervals)


def forward_fill(
    data: PersonIntervals, observation_window: TimeRange | None = None
) -> PersonIntervals:
    """
    Forward fills the intervals in the DataFrame.

    Each interval is extended to the next interval of a _different_ type.
    I.e. if there are multiple consecutive intervals of the same type, they are merged into one interval, and the
    last interval of these is extended until the start of the next interval (which is of a different type).

    If observation_window is provided, the last interval of each person is extended until the end of the
    observation window. If no observation_window is provided, the last interval of each person is not extended.

    :param data: The intervals per person.
    :param observation_window: The observation window.
    :return: A DataFrame with the forward filled intervals.
    """
    # return {person_id: interval.ffill() for person_id, interval in data.items()}

    result = {}

    for person_id, intervals in data.items():
        result[person_id] = forward_fill_intervals(intervals)

        if observation_window is not None:
            last_interval = result[person_id][-1]
            if last_interval.upper < observation_window.end.timestamp():
                result[person_id][-1] = Interval(
                    last_interval.lower,
                    observation_window.end.timestamp(),
                    last_interval.type,
                )

    return result


def complement_intervals(
    intervals: list[Interval],
    type_: IntervalType,
) -> list[Interval]:
    """
    Complements the intervals in the DataFrame.

    The complement of an interval is the interval that is not covered by the original interval.
    The complement of an interval is always of the same type as the original interval.

    :param intervals: The intervals per person.
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
            _impl.intersect_interval_lists(
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
            if operator == _impl.intersect_interval_lists:
                # if the operation is intersection, an empty dataframe means that the result is empty
                return dict()
            else:
                # if the operation is union, an empty dataframe can be ignored
                continue

        for group_keys, intervals in arr.items():
            intervals = _impl.union_rects(intervals)
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
    return _process_intervals(data, _impl.union_interval_lists)


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
            intervals_with_count = _impl.union_rects_with_count(intervals_with_count)
            if group_keys not in result:
                result[group_keys] = intervals_with_count
            else:
                result[group_keys] = _impl.union_rects_with_count(
                    result[group_keys] + intervals_with_count
                )

    return result


def filter_count_intervals(
    data: PersonIntervalsWithCount,
    min_count: int | None,
    max_count: int | None,
    keep_no_data: bool = True,
) -> PersonIntervals:
    """
    Filters the intervals per dict key in the list by count.

    :param data: A list of dict of intervals.
    :param min_count: The minimum count of the intervals.
    :param max_count: The maximum count of the intervals.
    :param keep_no_data: Whether to keep NO_DATA intervals (irrespective of the count).
    :return: A dict with the unioned intervals.
    """

    result: PersonIntervals = {}

    interval_filter = []
    if keep_no_data:
        interval_filter.append(IntervalType.NO_DATA)

    if min_count is None and max_count is None:
        raise ValueError("min_count and max_count cannot both be None")
    elif min_count is not None and max_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if min_count <= interval.count <= max_count
                or interval.type in interval_filter
            ]
    elif min_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if min_count <= interval.count or interval.type in interval_filter
            ]
    elif max_count is not None:
        for person_id in data:
            result[person_id] = [
                Interval(interval.lower, interval.upper, interval.type)
                for interval in data[person_id]
                if interval.count <= max_count or interval.type in interval_filter
            ]

    return result


def intersect_intervals(data: list[PersonIntervals]) -> PersonIntervals:
    """
    Intersects the intervals per dict key in the list.

    :param data: A list of dict of intervals.
    :return: A dict with the intersected intervals.
    """
    data = filter_dicts_by_common_keys(data)

    result = _process_intervals(data, _impl.intersect_interval_lists)

    return result


# ok
def mask_intervals(
    data: PersonIntervals,
    mask: PersonIntervals,
) -> PersonIntervals:
    """
    Masks the intervals in the dict per key.

    The intervals in data are intersected with the intervals in mask on a key-wise basis.
    The intervals outside the mask are removed.

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
        person_result = _impl.intersect_interval_lists(
            data[person_id], person_mask[person_id]
        )
        if not person_result:
            continue

        result[person_id] = person_result

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


def create_time_intervals(
    start_datetime: datetime.datetime | pendulum.DateTime,
    end_datetime: datetime.datetime | pendulum.DateTime,
    start_time: datetime.time,
    end_time: datetime.time,
    interval_type: IntervalType,
    timezone: pytz.tzinfo.DstTzInfo | str,
) -> list[Interval]:
    """
    Constructs a list of time intervals within a specified date range, each defined by daily start and end times.

    This function generates intervals for each day between the start and end datetimes, using specified start and end
    times to define each interval's boundaries. If the end time is earlier than the start time, the interval is assumed
    to span midnight. The function handles timezone differences and ensures that all calculations respect the time zone
    of the start datetime if provided, otherwise it uses the local timezone.

    If an interval is not fully contained within the specified date range, it is adjusted to fit within the boundaries.

    Parameters:
        start_datetime (datetime.datetime): The starting point of the date range for which intervals are created.
        end_datetime (datetime.datetime): The ending point of the date range.
        start_time (datetime.time): The daily start time for each interval.
        end_time (datetime.time): The daily end time for each interval, which may be on the following day if earlier
            than the start time.
        interval_type (IntervalType): The type of intervals to be created, which could denote the purpose or nature of
            the intervals.
        timezone (pytz.timezone): The timezone to use for the calculations.

    Returns:
        list[Interval]: A list of Interval namedtuples, each representing a time interval within the specified date
            range, with its start and end timestamps and the specified type.

    Raises:
        ValueError: If start_datetime and end_datetime are not in the same timezone.
    """
    if isinstance(timezone, str):
        timezone = cast(pytz.tzinfo.DstTzInfo, pytz.timezone(timezone))

    # todo: do not use these checks, there must be a universal way
    if isinstance(start_datetime, datetime.datetime) and not isinstance(
        start_datetime, pendulum.DateTime
    ):
        if (
            start_datetime.tzinfo is None
            or start_datetime.tzinfo.utcoffset(start_datetime) is None
        ):
            start_datetime = timezone.localize(start_datetime)
        else:
            start_datetime = start_datetime.astimezone(timezone)
        if (
            end_datetime.tzinfo is None
            or end_datetime.tzinfo.utcoffset(end_datetime) is None
        ):
            end_datetime = timezone.localize(end_datetime)
        else:
            end_datetime = end_datetime.astimezone(timezone)
    elif isinstance(start_datetime, pendulum.DateTime):
        assert isinstance(end_datetime, pendulum.DateTime)  # for mypy...
        start_datetime = start_datetime.in_timezone(timezone)
        end_datetime = end_datetime.in_timezone(timezone)

    # Prepare to collect intervals
    intervals = []
    previous_end = None
    def add_interval(interval_start, interval_end, interval_type):
        nonlocal previous_end
        effective_start = max(interval_start, start_datetime)
        effective_end = min(interval_end, end_datetime)
        if effective_start < effective_end:
            # We assert that the end point of the previous interval is
            # properly below (not just below or equal) the start of
            # the new interval because open/close events are generated
            # from these intervals by sorting (using a non-stable
            # method) which can result in an incorrect event order for
            # touching intervals.
            if previous_end is not None:
                assert previous_end < effective_start
            intervals.append(Interval(
                lower=effective_start.timestamp(),
                upper=effective_end.timestamp(),
                type=interval_type,
            ))
            previous_end = effective_end

    # Current date to process
    current_date = start_datetime.date()

    # Loop over each day from start_datetime to end_datetime
    while current_date <= end_datetime.date():
        # Calculate the datetime for the start time on the current date
        start_interval = timezone.localize(
            datetime.datetime.combine(current_date, start_time)
        )

        # Determine if the end time is on the next day
        if end_time <= start_time:
            end_interval = timezone.localize(
                datetime.datetime.combine(
                    current_date + datetime.timedelta(days=1), end_time
                )
            )
        else:
            end_interval = timezone.localize(
                datetime.datetime.combine(current_date, end_time)
            )

        # Create the interval with the specified interval_type if it
        # overlaps the main datetime range, otherwise fill the day
        # with an interval of type "not applicable".
        if end_interval < start_datetime: # completely before datetime range
            day_start = timezone.localize(
                datetime.datetime.combine(
                    current_date, datetime.time(0, 0, 0)
                ))
            day_end = timezone.localize(
                datetime.datetime.combine(
                    current_date, datetime.time(23, 59, 59)
                ))
            if (previous_end is not None) and day_start <= previous_end:
                start = previous_end + datetime.timedelta(seconds=1)
            else:
                start = day_start
            add_interval(start, day_end, IntervalType.NOT_APPLICABLE)
        elif end_datetime < start_interval: # completely after datetime range
            day_start = timezone.localize(
                datetime.datetime.combine(
                    current_date, datetime.time(0, 0, 0)
                ))
            if (previous_end is not None) and day_start <= previous_end:
                start = previous_end + datetime.timedelta(seconds=1)
            else:
                start = day_start
            add_interval(start, end_datetime, IntervalType.NOT_APPLICABLE)
        else:
            # Ensure the start of the interval is not before start_datetime and end of interval is not after end_datetime
            add_interval(start_interval, end_interval, interval_type)

        # Move to the next day
        current_date += datetime.timedelta(days=1)

    return intervals


def find_overlapping_windows(
    windows: list[Interval], data: PersonIntervals
) -> PersonIntervals:
    """
    Returns a list of windows that overlap with any interval in the intervals list. A window is included in the
    result if it overlaps in any part with any of the given intervals, not just where they intersect. The entire
    window is returned, not just the overlapping segment.

    Note that a single, common list of windows is used for all persons.

    :param windows: A list of windows, where each window is defined as an interval.
    :param data: The dict with intervals that are checked for overlap with the windows.
    :return: A list of windows that have any overlap with the intervals.
    """
    return {key: _impl.find_overlapping_windows(windows, data[key]) for key in data}


def find_overlapping_personal_windows(
    windows: PersonIntervals, data: PersonIntervals
) -> PersonIntervals:
    """
    Returns any windows (per person) that overlap with intervals in data.
    Unlike 'mask_intervals' which returns the intersection, this function
    returns the entire window from `windows` if it overlaps in any part
    with `data`.

    :param windows: A dict {person_id -> list of intervals (windows)}.
    :param data: A dict {person_id -> list of intervals to check against}.
    :return: A dict {person_id -> list of windows that overlap with data[person_id]}.
    """
    result = {}

    # Iterate over each person in `windows`
    for person_id, person_windows in windows.items():
        if person_id not in data:
            continue  # skip persons without data intervals

        # We collect all windows that overlap with any data intervals
        overlapping_windows = []
        for window in person_windows:
            # Check overlap with the intervals in data[person_id]
            # If intersection is non-empty for any data interval, we keep the entire window
            intersection = _impl.intersect_interval_lists([window], data[person_id])
            if intersection:
                overlapping_windows.append(window)

        if overlapping_windows:
            result[person_id] = overlapping_windows

    return result

def find_rectangles_with_count(data: list[PersonIntervals]) -> PersonIntervals:
    if len(data) == 0:
        return {}
    else:
        keys = data[0].keys()
        return {key: _impl.find_rectangles_with_count([ intervals[key] for intervals in data ])
                for key in keys}
