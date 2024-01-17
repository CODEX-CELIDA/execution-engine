import datetime
from operator import and_, or_
from typing import Callable, cast

from portion import Bound
from sqlalchemy import CursorResult

from execution_engine.task.process import Interval
from execution_engine.util.interval import (
    Atomic,
    DateTimeInterval,
    IntervalType,
    IntervalWithType,
)
from execution_engine.util.interval import empty_interval_datetime as empty_interval
from execution_engine.util.interval import interval_datetime as interval
from execution_engine.util.types import TimeRange

PersonIntervals = dict[int, IntervalWithType]


def normalize_interval(
    interval: DateTimeInterval,
) -> tuple[datetime.datetime, datetime.datetime, IntervalType]:
    """
    Normalizes the interval for storage in database.

    :param interval: The interval to normalize.
    :return: A tuple with the normalized interval.
    """

    return Interval(interval.lower, interval.upper, interval.type)


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
                result[group_keys] = result[group_keys] | intervals

    return result


def _interval_union(intervals: list[IntervalWithType]) -> IntervalWithType:
    """
    Performs a union on the given intervals.

    :param intervals: A list of intervals.
    :return: A list of intervals.
    """
    r = empty_interval()

    for interv in intervals:
        r |= interv

    return r


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
    return {person_id: interval.ffill() for person_id, interval in data.items()}


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
    baseline_interval = interval(
        observation_window.start, observation_window.end, interval_type_missing_persons
    )

    result = {}
    # invert intervals in time to get the missing intervals of existing persons

    for key in data:
        result[key] = (
            # take the least of the intersection of the observation window to retain the type of the
            #   original interval
            observation_window.interval(IntervalType.least_intersection_priority())
            & data[key].complement(type_=interval_type)
        )

    # get intervals for missing persons (in df w.r.t. base)
    missing_keys = set(reference) - set(result)

    for key in missing_keys:
        result[key] = baseline_interval

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

    return {person_id: intervals.invert_type() for person_id, intervals in data.items()}


def result_to_intervals(result: CursorResult) -> PersonIntervals:
    """
    Converts the result of the interval operations to a list of intervals.
    """
    person_interval = {}

    for row in result:
        if row.person_id not in person_interval:
            person_interval[row.person_id] = [
                Atomic(
                    Bound.CLOSED,
                    row.interval_start,
                    row.interval_end,
                    Bound.CLOSED,
                    row.interval_type,
                )
            ]
        else:
            person_interval[row.person_id].append(
                Atomic(
                    Bound.CLOSED,
                    row.interval_start,
                    row.interval_end,
                    Bound.CLOSED,
                    row.interval_type,
                )
            )

    for person_id in person_interval:
        person_interval[person_id] = DateTimeInterval(*person_interval[person_id])

    return cast(
        PersonIntervals, person_interval
    )  # person_intervals is a dict of person_id -> DateTimeInterval


def select_type(intervals: PersonIntervals, type_: IntervalType) -> PersonIntervals:
    """
    Selects the intervals of the given type.

    :param intervals: The intervals.
    :param type_: The type.
    :return: The intervals of the given type.
    """
    return {person_id: ival.select_type(type_) for person_id, ival in intervals.items()}


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
            if operator == and_:
                # if the operation is intersection, an empty dataframe means that the result is empty
                return dict()
            else:
                # if the operation is union, an empty dataframe can be ignored
                continue

        for group_keys, intervals in arr.items():
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
    return _process_intervals(data, or_)


def intersect_intervals(data: list[PersonIntervals]) -> PersonIntervals:
    """
    Intersects the intervals  per dict key in the list.

    :param data: A list of dict of intervals.
    :return: A dict with the intersected intervals.
    """
    data = filter_dicts_by_common_keys(data)
    result = _process_intervals(data, and_)

    return result


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
        person_id: intervals.replace(type_=IntervalType.least_intersection_priority())
        for person_id, intervals in mask.items()
    }

    result = {}
    for person_id in data:
        result[person_id] = data[person_id] & person_mask[person_id]

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
