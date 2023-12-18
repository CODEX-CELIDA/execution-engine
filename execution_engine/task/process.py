from typing import Callable

import pandas as pd

from execution_engine.util import TimeRange
from execution_engine.util.interval import DateTimeInterval as Interval
from execution_engine.util.interval import empty_interval_datetime as empty_interval
from execution_engine.util.interval import interval_datetime as interval


def interval_union(intervals: list[Interval]) -> Interval:
    """
    Performs a union on the given intervals.

    :param intervals: A list of intervals.
    :return: A list of intervals.
    """
    r = empty_interval()

    for interv in intervals:
        r |= interv

    return r


def insert_missing_intervals(
    df: pd.DataFrame, base: pd.DataFrame, by: list[str], observation_window: TimeRange
) -> pd.DataFrame:
    """
    Inserts the missing intervals in the DataFrame.

    :param df: The DataFrame with the intervals.
    :param base: The DataFrame with the base criterion.
    :param by: A list of column names to group by.
    :param observation_window: The observation window.
    :return: A DataFrame with the inserted intervals.
    """

    rows_df = df[by].drop_duplicates()
    rows_base = base[by].drop_duplicates()

    merged_df = pd.merge(rows_base, rows_df, on=by, how="outer", indicator=True)

    unique_to_base = merged_df[merged_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    unique_to_base = unique_to_base.assign(
        interval_start=observation_window.start, interval_end=observation_window.end
    )

    return pd.concat([df, unique_to_base])


def invert_intervals(
    df: pd.DataFrame, by: list[str], observation_window: TimeRange
) -> pd.DataFrame:
    """
    Inverts the intervals in the DataFrame.
    """

    # todo: do we need to list all persons here, because inverting an empty set should yield
    #  the full set but we do not know about empty sets here (as they are not in the df)?

    result = {}
    for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
        new_intervals = to_intervals(group[["interval_start", "interval_end"]])
        new_interval_union = interval_union(new_intervals)
        result[group_keys] = (
            observation_window.interval() & new_interval_union.complement()
        )

    return _result_to_df(result, by)


def to_intervals(df: pd.DataFrame) -> list[Interval]:
    """
    Converts the DataFrame to intervals.

    :param df: A DataFrame with columns "interval_start" and "interval_end".
    :return: A list of intervals.
    """

    return [
        interval(start, end)
        for start, end in zip(df["interval_start"], df["interval_end"])
    ]


def timestamps_to_intervalsX(group: pd.DataFrame) -> list[Interval]:
    """
    Converts the timestamps in the DataFrame to intervals.

    :param group: A DataFrame with columns "interval_start" and "interval_end" containing timestamps.
    :return: A list of intervals.
    """
    divisor = 1000000000
    group = group.astype("int64") // divisor
    return [
        interval(start, end)
        for start, end in zip(group["interval_start"], group["interval_end"])
    ]


def _process_intervals(
    dfs: list[pd.DataFrame], by: list[str], operation: Callable
) -> pd.DataFrame:
    """
    Processes the intervals in the DataFrames (intersect or union)

    :param dfs: A list of DataFrames.
    :param by: A list of column names to group by.
    :param operation: The operation to perform on the intervals (intersect or union).
    :return: A DataFrame with the processed intervals.
    """
    # assert dfs is a list of dataframes
    assert isinstance(dfs, list) and all(
        isinstance(df, pd.DataFrame) for df in dfs
    ), "dfs must be a list of DataFrames"

    result = {}
    for df in dfs:
        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = to_intervals(group[["interval_start", "interval_end"]])
            new_interval_union = interval_union(new_intervals)

            if group_keys not in result:
                result[group_keys] = new_interval_union
            else:
                result[group_keys] = operation(result[group_keys], new_interval_union)

    return _result_to_df(result, by)


def merge_intervals(dfs: list[pd.DataFrame], by: list[str]) -> pd.DataFrame:
    """
    Merges the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :param by: A list of column names to group by.
    :return: A DataFrame with the merged intervals.
    """

    return _process_intervals(dfs, by, lambda x, y: x | y)


def intersect_intervals(dfs: list[pd.DataFrame], by: list[str]) -> pd.DataFrame:
    """
    Intersects the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :param by: A list of column names to group by.
    :return: A DataFrame with the intersected intervals.
    """
    dfs = filter_common_items(dfs, by)

    return _process_intervals(dfs, by, lambda x, y: x & y)


def _result_to_df(
    result: dict[tuple[str] | str, Interval], by: list[str]
) -> pd.DataFrame:
    """
    Converts the result of the interval operations to a DataFrame.

    :param result: The result of the interval operations.
    :param by: A list of column names to group by.
    :return: A DataFrame with the interval results.
    """
    records = []
    for group_keys, intervals in result.items():
        # Check if group_keys is a tuple or a single value and unpack accordingly
        if isinstance(group_keys, tuple):
            record_keys = dict(zip(by, group_keys))
        else:
            record_keys = {by[0]: group_keys}

        for interv in intervals:
            record = {
                **record_keys,
                "interval_start": interv.lower,
                "interval_end": interv.upper,
            }
            records.append(record)

    return pd.DataFrame(records)


def filter_common_items(
    dfs: list[pd.DataFrame], columns: list[str]
) -> list[pd.DataFrame]:
    """
    Filters the DataFrames based on common items in the specified columns.

    Returned are only those rows of each dataframe of which the values in the columns identified
    by the parameter `columns` are common to all dataframes.

    :param dfs: A list of DataFrames.
    :param columns: A list of column names to filter on.
    :return: A list of DataFrames with the common items.
    """

    def unique_items(df: pd.DataFrame) -> set[tuple]:
        return set(tuple(row) for row in df.itertuples(index=False))

    common_items = unique_items(dfs[0][columns])

    # Find common items across all DataFrames
    for df in dfs[1:]:
        common_items.intersection_update(unique_items(df[columns]))

    filtered_dfs = []
    for df in dfs:
        # Filter DataFrame based on common items
        filtered_df = df[
            df.apply(lambda row: tuple(row[columns]) in common_items, axis=1)
        ]
        filtered_dfs.append(filtered_df)

    return filtered_dfs
