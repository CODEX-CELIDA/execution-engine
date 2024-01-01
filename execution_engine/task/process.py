from functools import reduce
from typing import Callable

import pandas as pd

from execution_engine.util import TimeRange
from execution_engine.util.interval import DateTimeInterval as Interval
from execution_engine.util.interval import IntervalType
from execution_engine.util.interval import empty_interval_datetime as empty_interval
from execution_engine.util.interval import interval_datetime as interval

df_dtypes = {
    "person_id": "int64",
    "interval_start": "datetime64[ns, UTC]",
    "interval_end": "datetime64[ns, UTC]",
    "interval_type": "category",
}


def _or_(x: Interval, y: Interval) -> Interval:
    """
    Performs a union on the given intervals.

    Helper function for _process intervals.
    """
    return x | y


def _and_(x: Interval, y: Interval) -> Interval:
    """
    Performs am intersection on the given intervals.

    Helper function for _process intervals.
    """

    return x & y


def concat_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates the DataFrames in the list.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the concatenated DataFrames.
    """
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return pd.DataFrame(columns=df_dtypes.keys())
    return pd.concat(dfs)


def unique_items(df: pd.DataFrame) -> set[tuple]:
    """
    Returns the unique items in the DataFrame.
    """
    return set(df.itertuples(index=False, name=None))


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


def complementary_intervals(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    observation_window: TimeRange,
    interval_type: IntervalType,
) -> pd.DataFrame:
    """
    Insert missing intervals into a dataframe, i.e. the complement of the existing intervals w.r.t. the observation
    window. Persons that are not in the dataframe but in the base dataframe are added as well (with the full
    observation window as the interval).

    :param df: The DataFrame with the intervals.
    :param reference_df: The DataFrame with the base criterion. Used to determine the patients for which intervals
        are missing.
    :param observation_window: The observation window.
    :param interval_type: The type of the complementary intervals for persons existing in the DataFrame.
    :return: A DataFrame with the inserted intervals.
    """
    by = ["person_id"]

    interval_type_missing_persons = interval_type

    # get intervals for missing persons (in df w.r.t. base)
    rows_df = df[by].drop_duplicates()
    rows_base = reference_df[by].drop_duplicates()

    merged_df = pd.merge(rows_base, rows_df, on=by, how="outer", indicator=True)

    df_missing_persons = (
        merged_df[merged_df["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .assign(
            interval_start=observation_window.start,
            interval_end=observation_window.end,
            interval_type=interval_type_missing_persons,
        )
    )

    # invert intervals in time to get the missing intervals of existing persons
    if not df.empty:
        result = {}

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = df_to_intervals(
                group[["interval_start", "interval_end", "interval_type"]]
            )
            new_interval_union = _interval_union(new_intervals)

            result[group_keys] = (
                # take the least of the intersection of the observation window to retain the type of the
                #   original interval
                observation_window.interval(IntervalType.least_intersection_priority())
                & new_interval_union.complement(type_=interval_type)
            )

        df = _result_to_df(result, by)

    return concat_dfs([df, df_missing_persons])


def invert_intervals(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    observation_window: TimeRange,
) -> pd.DataFrame:
    """
    Inverts the intervals in the DataFrame.

    The inversion is performed in time and value, i.e. first the complement set of the intervals is taken (w.r.t. the
    observation window) and then the original intervals are added back in, but with the inverted value.

    Also, the full observation_window is added for all patients in base that are not in df.

    :param df: The DataFrame with the intervals.
    :param reference_df: The DataFrame with the base criterion.
    :param observation_window: The observation window.
    :return: A DataFrame with the inserted intervals.
    """
    df_c = complementary_intervals(
        df,
        reference_df=reference_df,
        observation_window=observation_window,
        interval_type=IntervalType.NEGATIVE,  # undefined intervals are always considered negative; criteria must explicitly set NO_DATA
    )

    df = concat_dfs([df, df_c])

    df["interval_type"] = ~df["interval_type"]

    return df


def df_to_intervals(df: pd.DataFrame) -> list[Interval]:
    """
    Converts the DataFrame to intervals.

    :param df: A DataFrame with columns "interval_start" and "interval_end".
    :return: A list of intervals.
    """

    return [
        interval(start, end, type_)
        for start, end, type_ in zip(
            df["interval_start"], df["interval_end"], df["interval_type"]
        )
    ]


def _process_intervals(dfs: list[pd.DataFrame], operator: Callable) -> pd.DataFrame:
    """
    Processes the intervals in the DataFrames (intersect or union)

    :param dfs: A list of DataFrames.
    :param operator: The operation to perform on the intervals (intersect or union).
    :return: A DataFrame with the processed intervals.
    """

    by = ["person_id"]

    if not len(dfs):
        return dfs

    # assert dfs is a list of dataframes
    assert isinstance(dfs, list) and all(
        isinstance(df, pd.DataFrame) for df in dfs
    ), "dfs must be a list of DataFrames"

    result = {}

    for df in dfs:
        if df.empty:
            if operator == _and_:
                # if the operation is intersection, an empty dataframe means that the result is empty
                return pd.DataFrame(columns=df.columns)
            else:
                # if the operation is union, an empty dataframe can be ignored
                continue

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = df_to_intervals(
                group[["interval_start", "interval_end", "interval_type"]]
            )
            new_interval_union = _interval_union(new_intervals)

            if group_keys not in result:
                result[group_keys] = new_interval_union
            else:
                result[group_keys] = operator(result[group_keys], new_interval_union)

    return _result_to_df(result, by)


def union_intervals(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the merged intervals.
    """

    df = _process_intervals(dfs, _or_)
    # df = merge_interval_across_types(df, operator='OR')

    return df


def intersect_intervals(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Intersects the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the intersected intervals.
    """
    dfs = filter_dataframes_by_shared_column_values(dfs, columns=["person_id"])
    df = _process_intervals(dfs, _and_)

    return df


def mask_intervals(
    df: pd.DataFrame,
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Masks the intervals in the DataFrames grouped by columns.

    The intervals in df are intersected with the intervals in mask. The intervals outside the mask are set to
    interval_type_outside_mask, the intervals inside the mask are left unchanged.

    :param df: A DataFrames with intervals that should be masked.
    :param mask: A DataFrame with intervals that should be used for masking.
    :return: A DataFrame with the masked intervals.
    """

    by = ["person_id"]

    person_mask = {}
    for person_id, group in mask.groupby(by=["person_id"]):
        # set to the least priority to retain type of original interval
        # todo: can this be made more efficient?
        group = group.assign(interval_type=IntervalType.least_intersection_priority())
        person_mask[person_id[0]] = _interval_union(
            df_to_intervals(group[["interval_start", "interval_end", "interval_type"]])
        )

    result = {}
    for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
        intervals = df_to_intervals(
            group[["interval_start", "interval_end", "interval_type"]]
        )
        intervals = _interval_union(intervals)

        result[group_keys] = intervals & person_mask[group_keys[by.index("person_id")]]

    result = _result_to_df(result, by)

    return result


# todo: remove me
def filter_dataframes_by_shared_column_valuesX(
    dfs: list[pd.DataFrame], columns: list[str]
) -> list[pd.DataFrame]:
    """
    Filters the DataFrames based on shared values in the specified columns.

    Returned are only those rows of each dataframe where the values in the columns identified
    by the parameter `columns` are shared across all dataframes.

    :param dfs: A list of DataFrames.
    :param columns: A list of column names to filter on.
    :return: A list of DataFrames with the rows that have shared column values.
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


def filter_dataframes_by_shared_column_values(
    dfs: list[pd.DataFrame], columns: list[str]
) -> list[pd.DataFrame]:
    """
    Filters the DataFrames based on shared values in the specified columns.

    Returned are only those rows of each dataframe where the values in the columns identified
    by the parameter `columns` are shared across all dataframes.

    :param dfs: A list of DataFrames.
    :param columns: A list of column names to filter on.
    :return: A list of DataFrames with the rows that have shared column values.
    """

    if len(dfs) == 1:
        return dfs

    # Find common rows across all DataFrames
    # Use reduce to iteratively inner join DataFrames on the specified columns
    common_rows = reduce(
        lambda left, right: pd.merge(left[columns], right[columns], on=columns), dfs
    )

    # Drop duplicate rows to keep only unique combinations of the specified columns
    common_rows = common_rows.drop_duplicates()

    # Filter each DataFrame to keep only the common rows
    filtered_dfs = []

    for df in dfs:
        # Merge with common_rows and keep only the rows present in common_rows
        filtered_df = pd.merge(df, common_rows, on=columns, how="inner")
        filtered_dfs.append(filtered_df)

    return filtered_dfs


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
                "interval_type": interv.type,
            }
            records.append(record)

    return pd.DataFrame(
        records, columns=by + ["interval_start", "interval_end", "interval_type"]
    )
