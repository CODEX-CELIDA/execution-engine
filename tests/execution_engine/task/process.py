import pandas as pd
import pytz

from execution_engine.task.process import (
    _result_to_df,
    filter_common_items,
    intersect_intervals,
    invert_intervals,
    merge_intervals,
    timestamps_to_intervals,
)
from execution_engine.util import TimeRange
from execution_engine.util.interval import interval


def test_timestamps_to_intervals():
    # Create sample data
    data = {
        "interval_start": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "interval_end": pd.to_datetime(["2023-01-02", "2023-01-03"]),
    }
    df = pd.DataFrame(data)

    # Call the function
    result = timestamps_to_intervals(df)

    # Expected intervals
    expected = [interval(1672531200, 1672617600), interval(1672617600, 1672704000)]

    # Assert the result
    assert result == expected


def test_invert():
    # Create sample data
    data = {
        "person_id": [1, 2, 1, 2],
        "concept_id": ["A", "A", "A", "A"],
        "interval_start": pd.to_datetime(
            [
                "2023-01-01T00:00:00",
                "2023-01-01T12:00:00",
                "2023-01-01T06:00:00",
                "2023-01-02T12:00:00",
            ]
        ),
        "interval_end": pd.to_datetime(
            [
                "2023-01-01T12:00:00",
                "2023-01-02T06:00:00",
                "2023-01-01T18:00:00",
                "2023-01-03T12:00:00",
            ]
        ),
    }
    df = pd.DataFrame(data)
    by = ["person_id", "concept_id"]
    observation_window = TimeRange(
        name="observation", start="2023-01-01 00:00:00Z", end="2023-01-02 18:00:00Z"
    )

    # Call the function
    result = invert_intervals(df, by, observation_window)

    # Expected data
    expected_data = {
        "person_id": [1, 2, 2],
        "concept_id": ["A", "A", "A"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T18:00:00", "2023-01-01T0:00:00", "2023-01-02T06:00:00"]
        ),
        "interval_end": pd.to_datetime(
            [
                "2023-01-02T18:00:00",
                "2023-01-01T12:00:00",
                "2023-01-02T12:00:00",
            ]
        ),
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert the result
    pd.testing.assert_frame_equal(result, expected_df)


def test_result_to_df_single_key():
    # Test with single key and specific timezones
    result = {1: [interval(1609459200, 1609484400), interval(1609545600, 1609570800)]}
    by = ["person_id"]
    tz_start = pytz.timezone("UTC")
    tz_end = pytz.timezone("America/New_York")

    expected_data = {
        "person_id": [1, 1],
        "interval_start": [
            pd.Timestamp("2021-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2021-01-02 00:00:00", tz="UTC"),
        ],
        "interval_end": [
            pd.Timestamp("2021-01-01 02:00:00", tz="America/New_York"),
            pd.Timestamp("2021-01-02 02:00:00", tz="America/New_York"),
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    # Call the function
    result_df = _result_to_df(result, by, tz_start, tz_end)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_result_to_df_multiple_keys():
    # Test with multiple keys and None for timezones
    result = {
        (1, "A"): [interval(1609459200, 1609484400)],
        (2, "B"): [interval(1609545600, 1609570800)],
    }
    by = ["person_id", "concept_id"]
    tz_start = "UTC"
    tz_end = "America/New_York"

    expected_data = {
        "person_id": [1, 2],
        "concept_id": ["A", "B"],
        "interval_start": [
            pd.Timestamp("2021-01-01 00:00:00", tz=tz_start),
            pd.Timestamp("2021-01-02 00:00:00", tz=tz_start),
        ],
        "interval_end": [
            pd.Timestamp("2021-01-01 02:00:00", tz=tz_end),
            pd.Timestamp("2021-01-02 02:00:00", tz=tz_end),
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    # Call the function
    result_df = _result_to_df(result, by, tz_start, tz_end)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_merge_intervals():
    # Prepare test data with datetime64[ns, UTC] dtype
    data1 = {
        "person_id": [1, 1],
        "concept_id": ["A", "A"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"], utc=True
        ),
        "interval_end": pd.to_datetime(
            ["2023-01-01T12:00:00Z", "2023-01-02T12:00:00Z"], utc=True
        ),
    }
    data2 = {
        "person_id": [1, 1],
        "concept_id": ["A", "B"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T06:00:00Z", "2023-01-03T00:00:00Z"], utc=True
        ),
        "interval_end": pd.to_datetime(
            ["2023-01-01T18:00:00Z", "2023-01-03T12:00:00Z"], utc=True
        ),
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Call the function
    result_df = merge_intervals([df1, df2], ["person_id", "concept_id"])

    # Define expected output
    expected_data = {
        "person_id": [1, 1, 1],
        "concept_id": ["A", "A", "B"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z", "2023-01-03T00:00:00Z"],
            utc=True,
        ),
        "interval_end": pd.to_datetime(
            ["2023-01-01T18:00:00Z", "2023-01-02T12:00:00Z", "2023-01-03T12:00:00Z"],
            utc=True,
        ),
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_intersect_intervals():
    # Prepare test data with datetime64[ns, UTC] dtype
    data1 = {
        "person_id": [1, 1],
        "concept_id": ["A", "A"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"], utc=True
        ),
        "interval_end": pd.to_datetime(
            ["2023-01-01T12:00:00Z", "2023-01-02T12:00:00Z"], utc=True
        ),
    }
    data2 = {
        "person_id": [1, 1],
        "concept_id": ["A", "B"],
        "interval_start": pd.to_datetime(
            ["2023-01-01T06:00:00Z", "2023-01-03T00:00:00Z"], utc=True
        ),
        "interval_end": pd.to_datetime(
            ["2023-01-01T18:00:00Z", "2023-01-03T12:00:00Z"], utc=True
        ),
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Call the function
    result_df = intersect_intervals([df1, df2], ["person_id", "concept_id"])

    # Define expected output
    expected_data = {
        "person_id": [1],
        "concept_id": ["A"],
        "interval_start": pd.to_datetime(["2023-01-01T06:00:00Z"], utc=True),
        "interval_end": pd.to_datetime(["2023-01-01T12:00:00Z"], utc=True),
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_filter_common_items():
    # Create sample DataFrames
    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": ["a1", "a2", "a3"],
        }
    )
    df2 = pd.DataFrame(
        {
            "A": [3, 4, 5],
            "B": ["z", "z", "w"],
            "C": ["b1", "b2", "b3"],
        }
    )
    df3 = pd.DataFrame(
        {
            "A": [1, 3, 6],
            "B": ["x", "z", "v"],
            "C": ["c1", "c2", "c3"],
        }
    )

    # Call the function
    filtered_dfs = filter_common_items([df1, df2, df3], ["A", "B"])

    # Define expected output
    expected_df1 = pd.DataFrame(
        {
            "A": [3],
            "B": ["z"],
            "C": ["a3"],
        },
        index=[2],
    )
    expected_df2 = pd.DataFrame(
        {
            "A": [3],
            "B": ["z"],
            "C": ["b1"],
        },
        index=[0],
    )
    expected_df3 = pd.DataFrame(
        {
            "A": [3],
            "B": ["z"],
            "C": ["c2"],
        },
        index=[1],
    )

    # Assert
    pd.testing.assert_frame_equal(filtered_dfs[0], expected_df1)
    pd.testing.assert_frame_equal(filtered_dfs[1], expected_df2)
    pd.testing.assert_frame_equal(filtered_dfs[2], expected_df3)
