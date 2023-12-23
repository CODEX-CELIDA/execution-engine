from typing import Callable

import pandas as pd

from execution_engine.constants import IntervalType
from execution_engine.util import TimeRange
from execution_engine.util.interval import DateTimeInterval as Interval
from execution_engine.util.interval import empty_interval_datetime as empty_interval
from execution_engine.util.interval import interval_datetime as interval


def unique_items(df: pd.DataFrame) -> set[tuple]:
    """
    Returns the unique items in the DataFrame.
    """
    return set(tuple(row) for row in df.itertuples(index=False))


def _interval_union(intervals: list[Interval]) -> Interval:
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
    df: pd.DataFrame,
    base: pd.DataFrame,
    observation_window: TimeRange,
    interval_type: IntervalType,
) -> pd.DataFrame:
    """
    Insert missing intervals into a dataframe, determined by the keys in the base dataframe.

    :param df: The DataFrame with the intervals.
    :param base: The DataFrame with the base criterion.
    :param observation_window: The observation window.
    :param interval_type: The type of the intervals that are added.
    :return: A DataFrame with the inserted intervals.
    """
    by = ["person_id"]

    rows_df = df[by].drop_duplicates()
    rows_base = base[by].drop_duplicates()

    merged_df = pd.merge(rows_base, rows_df, on=by, how="outer", indicator=True)

    unique_to_base = merged_df[merged_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    unique_to_base = unique_to_base.assign(
        interval_start=observation_window.start,
        interval_end=observation_window.end,
        interval_type=interval_type,
    )

    return pd.concat([df, unique_to_base])


def invert_intervals(
    df: pd.DataFrame,
    base: pd.DataFrame,
    by: list[str],
    observation_window: TimeRange,
    interval_type: IntervalType | None,
    missing_interval_type: IntervalType,
) -> pd.DataFrame:
    """
    Inverts the intervals in the DataFrame.

    Inserts the full observation_window for all patients in base that are not in df.

    :param df: The DataFrame with the intervals.
    :param base: The DataFrame with the base criterion.
    :param by: A list of column names to group by.
    :param observation_window: The observation window.
    :param interval_type: The type of the intervals that are added. None if it should be inferred from the DataFrame.
    :param missing_interval_type: The type of the newly added intervals that are missing after inversion.
    :return: A DataFrame with the inserted intervals.
    """

    if not len(by):
        raise ValueError("by must not be empty")

    if "interval_type" in by:
        raise ValueError("by must not contain interval_type")

    unique_persons = df["person_id"].unique()

    if not df.empty:
        result = {}

        if interval_type is None:
            assert (
                df.groupby(by)["interval_type"].nunique().nunique() == 1
            ), "only one interval_type per group is supported"
            interval_types = df.groupby(by)["interval_type"].first()

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = to_intervals(group[["interval_start", "interval_end"]])
            new_interval_union = _interval_union(new_intervals)

            result[group_keys] = (
                observation_window.interval() & new_interval_union.complement()
            )

        df = _result_to_df(result, by)

        if interval_type is None:
            df = pd.merge(df, interval_types, on="person_id")
        else:
            df["interval_type"] = interval_type

    return insert_missing_intervals(
        df,
        base=base[~base["person_id"].isin(unique_persons)],
        observation_window=observation_window,
        interval_type=missing_interval_type,
    )


def combine_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine overlapping intervals in a DataFrame based on a specified priority order.

    This function iterates through a sorted DataFrame of intervals and combines them based on the interval type's
    priority. The priority order is NEGATIVE > POSITIVE > NODATA. In the case of overlapping intervals with different
    types, the interval with the higher priority is chosen. If intervals of the same type overlap, they are merged into
    a single interval.

    Parameters:
    sorted_df (pd.DataFrame): A Pandas DataFrame containing the intervals. It should have the columns
                              'person_id', 'interval_start', 'interval_end', and 'interval_type'.
                              The DataFrame should be sorted as described above.

    Returns:
    pd.DataFrame: A new DataFrame with combined intervals as per the specified rules.
                   The resulting DataFrame will have the same columns as the input DataFrame.
    """
    # Define the custom sort order for 'interval_type'
    priority_order = [
        IntervalType.NEGATIVE,
        IntervalType.POSITIVE,
        IntervalType.NO_DATA,
        IntervalType.NOT_APPLICABLE,
    ]

    # Convert 'interval_type' to a Categorical type with the specified order
    df["interval_type"] = pd.Categorical(
        df["interval_type"], categories=priority_order, ordered=True
    )

    df = df.sort_values(
        by=["person_id", "interval_type", "interval_start"],
        ascending=[True, True, True],
    )
    combined: list[pd.Series] = []

    for _, row in df.iterrows():
        if not combined:
            combined.append(row)
            continue

        last = combined[-1]
        if (
            row["person_id"] == last["person_id"]
            and row["interval_start"] <= last["interval_end"]
        ):
            # Overlapping interval found
            if row["interval_end"] > last["interval_end"]:
                # Split the interval based on priority
                combined.append(row)
        else:
            combined.append(row)

    return pd.DataFrame(combined)


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


def timestamps_to_intervals(group: pd.DataFrame) -> list[Interval]:
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

    if not len(dfs):
        return dfs

    # assert dfs is a list of dataframes
    assert isinstance(dfs, list) and all(
        isinstance(df, pd.DataFrame) for df in dfs
    ), "dfs must be a list of DataFrames"

    result = {}

    for df in dfs:
        if df.empty:
            continue

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = to_intervals(group[["interval_start", "interval_end"]])
            new_interval_union = _interval_union(new_intervals)

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

    df = _process_intervals(dfs, by, lambda x, y: x | y)
    df = combine_intervals(df)

    return df


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

    return pd.DataFrame(records, columns=by + ["interval_start", "interval_end"])


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

    if not len(dfs):
        return dfs

    # make sure the columns exist in all dataframes, otherwise we get a KeyError - name the missing columns
    # in the error message
    for i, df in enumerate(dfs):
        for column in columns:
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame {i}")

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
