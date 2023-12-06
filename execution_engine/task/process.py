import datetime
from typing import Callable

import pandas as pd
from interval import interval

from execution_engine.util import TimeRange


def invert(df: pd.DataFrame, observation_window: TimeRange) -> pd.DataFrame:
    """
    Inverts the intervals in the DataFrame.
    """
    # Invert logic
    return df


def timestamps_to_intervals(group: pd.DataFrame) -> list[interval]:
    """
    Converts the timestamps in the DataFrame to intervals.

    :param group: A DataFrame with columns "interval_start" and "interval_end" containing timestamps.
    :return: A list of intervals.
    """
    group = group.astype("int64") / 1e9
    return [
        interval([start, end])
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
    result = {}
    for df in dfs:
        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = timestamps_to_intervals(
                group[["interval_start", "interval_end"]]
            )
            new_interval_union = interval.union(new_intervals)

            if group_keys not in result:
                result[group_keys] = new_interval_union
            else:
                result[group_keys] = operation(result[group_keys], new_interval_union)

    tz_start = df["interval_start"].dt.tz if not df["interval_start"].empty else None
    tz_end = df["interval_end"].dt.tz if not df["interval_end"].empty else None

    return _result_to_df(result, by, tz_start, tz_end)


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
    result: dict[tuple[str] | str, interval],
    by: list[str],
    tz_start: datetime.tzinfo | str | None,
    tz_end: datetime.tzinfo | str | None,
) -> pd.DataFrame:
    """
    Converts the result of the interval operations to a DataFrame.

    :param result: The result of the interval operations.
    :param by: A list of column names to group by.
    :param tz_start: The timezone of the interval start.
    :param tz_end: The timezone of the interval end.
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
            interval_start = pd.to_datetime(interv[0], unit="s", utc=True).tz_convert(
                tz_start
            )
            interval_end = pd.to_datetime(interv[1], unit="s", utc=True).tz_convert(
                tz_end
            )

            record = {
                **record_keys,
                "interval_start": interval_start,
                "interval_end": interval_end,
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
