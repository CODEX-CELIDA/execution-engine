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


def concat_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates the DataFrames in the list.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the concatenated DataFrames.
    """
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs)


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


def complementary_intervals(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    observation_window: TimeRange,
    interval_type: IntervalType | str = "auto",
    interval_type_missing_persons: IntervalType = IntervalType.NO_DATA,
) -> pd.DataFrame:
    """
    Insert missing intervals into a dataframe, i.e. the complement of the existing intervals w.r.t. the observation
    window. Persons that are not in the dataframe but in the base dataframe are added as well (with the full
    observation window as the interval).

    :param df: The DataFrame with the intervals.
    :param reference_df: The DataFrame with the base criterion. Used to determine the patients for which intervals
        are missing.
    :param observation_window: The observation window.
    :param interval_type: The type of the complementary intervals for persons existing in the DataFrame. If 'auto',
        the interval_type should be the (unique) interval_type of the person's intervals in the DataFrame.
    :param interval_type_missing_persons: The interval type for the complementary intervals for persons that are not
        in the DataFrame but only in the reference DataFrame.
    :return: A DataFrame with the inserted intervals.
    """
    by = ["person_id"]

    if not isinstance(interval_type, IntervalType) and not interval_type == "auto":
        raise ValueError(
            f"interval_type must be an IntervalType or 'auto', got {interval_type}"
        )

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

        if interval_type == "auto":
            assert (
                df.groupby(by)["interval_type"].nunique().nunique() == 1
            ), "only one interval_type per group is supported"
            interval_types = df.groupby(by)["interval_type"].first()

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = df_to_intervals(
                group[["interval_start", "interval_end", "interval_type"]]
            )
            new_interval_union = _interval_union(new_intervals)

            result[group_keys] = (
                # take the least of the intersection of the observation window to retain the type of the
                #   original interval
                observation_window.interval(IntervalType.least_intersection_priority())
                & new_interval_union.complement(
                    type_=interval_type if interval_type != "auto" else None  # type: ignore # type_ is not 'str' here (as mypy thinks)
                )
            )

        df = _result_to_df(result, by)

        # todo: remove me
        if interval_type == "auto":
            X = pd.merge(df, interval_types, on="person_id")
            assert X["interval_type_x"].equals(X["interval_type_y"])
            # todo: remove me
        else:
            df["interval_type"] = interval_type

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
        interval_type="auto",
    )
    df["interval_type"] = ~df["interval_type"]

    df = concat_dfs([df, df_c])

    return df


def merge_interval_across_types(df: pd.DataFrame, operator: str) -> pd.DataFrame:
    """
    Merge overlapping intervals in a DataFrame based on an operator (AND/OR) and the associated priority order of
    interval types.

    This function iterates through a sorted DataFrame of intervals and combines them based on the interval type's
    priority. The priority order is:
        -   for AND-combination: NEGATIVE > POSITIVE > NOT_APPLICABLE > NODATA
        -   for OR-combination: POSITIVE > NODATA > NOT_APPLICABLE > NEGATIVE
    In the case of overlapping intervals with different types, the interval with the higher priority is chosen.
    If intervals of the same type overlap, they are merged into a single interval.

    :param df: A Pandas DataFrame containing the intervals. It should have the columns
                  'person_id', 'interval_start', 'interval_end', and 'interval_type'.
    :param operator: The operator to use for combining the intervals. Valid values are 'AND' and 'OR'.
    :return: A new DataFrame with combined intervals as per the specified rules.
        The resulting DataFrame will have the same columns as the input DataFrame.
    """
    # Define the custom sort order for 'interval_type'
    if operator == "AND":
        priority_order = [
            IntervalType.NEGATIVE,
            IntervalType.POSITIVE,
            IntervalType.NOT_APPLICABLE,
            IntervalType.NO_DATA,
        ]
    elif operator == "OR":
        priority_order = [
            IntervalType.POSITIVE,
            IntervalType.NO_DATA,
            IntervalType.NOT_APPLICABLE,
            IntervalType.NEGATIVE,
        ]
    else:
        raise ValueError(f"Invalid operator '{operator}'")

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


def _process_intervals(dfs: list[pd.DataFrame], operation: Callable) -> pd.DataFrame:
    """
    Processes the intervals in the DataFrames (intersect or union)

    :param dfs: A list of DataFrames.
    :param operation: The operation to perform on the intervals (intersect or union).
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
            continue

        for group_keys, group in df.groupby(by, as_index=False, group_keys=False):
            new_intervals = df_to_intervals(
                group[["interval_start", "interval_end", "interval_type"]]
            )
            new_interval_union = _interval_union(new_intervals)

            if group_keys not in result:
                result[group_keys] = new_interval_union
            else:
                result[group_keys] = operation(result[group_keys], new_interval_union)

    return _result_to_df(result, by)


def union_intervals(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the merged intervals.
    """

    df = _process_intervals(dfs, lambda x, y: x | y)
    # df = merge_interval_across_types(df, operator='OR')

    return df


def intersect_intervals(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Intersects the intervals in the DataFrames grouped by columns.

    :param dfs: A list of DataFrames.
    :return: A DataFrame with the intersected intervals.
    """
    # dfs = filter_dataframes_by_shared_column_values(dfs, by)

    df = _process_intervals(dfs, lambda x, y: x & y)
    # df = merge_interval_across_types(df, operator='AND')

    return df


# todo: remove parameters once it's clear this functionality isn't required
def mask_intervals(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    # interval_type_outside_mask: IntervalType,
    # observation_window: TimeRange,
) -> pd.DataFrame:
    """
    Masks the intervals in the DataFrames grouped by columns.

    The intervals in df are intersected with the intervals in mask. The intervals outside the mask are set to
    interval_type_outside_mask, the intervals inside the mask are left unchanged.

    :param df: A DataFrames with intervals that should be masked.
    :param mask: A DataFrame with intervals that should be used for masking.
    :param interval_type_outside_mask: The interval type for intervals outside the mask.
    :param observation_window: The observation window.
    :return: A DataFrame with the masked intervals.
    """

    by = ["person_id"]

    person_mask = {}
    for person_id, group in mask.groupby(by=["person_id"]):
        # set to the least priority to retain type of original interval
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

    # todo: remove comment if this is really not needed
    # removed because this adds intervals for patients that do not have any intervals, but this
    #       is not really required (I think). it just adds a lot of data to the database.
    # df_c = complementary_intervals(
    #    result, mask, observation_window=observation_window, interval_type=interval_type_outside_mask
    # )
    # result = _concat_dfs([result, df_c])

    return result


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
