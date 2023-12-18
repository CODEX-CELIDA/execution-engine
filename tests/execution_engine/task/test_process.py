import datetime
from io import StringIO

import pandas as pd
import pendulum
import pytest
import pytz
from pytz.tzinfo import DstTzInfo

from execution_engine.task.process import (
    _interval_union,
    _result_to_df,
    filter_common_items,
    intersect_intervals,
    invert_intervals,
    merge_intervals,
)
from execution_engine.task.process import (
    timestamps_to_intervals as timestamps_to_intervals,
)
from execution_engine.task.process import to_intervals
from execution_engine.util import TimeRange
from execution_engine.util.interval import (
    DateTimeInterval,
    empty_interval_datetime,
    interval_datetime,
    interval_int,
)

# todo: test interval inversion
# todo: test interval and
# todo: test interval or


def interval(start: str, end: str) -> DateTimeInterval:
    return interval_datetime(pendulum.parse(start), pendulum.parse(end))


def parse_dt(s: str, tz: DstTzInfo) -> datetime.datetime:
    return tz.localize(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))


def df_from_str(data_str):
    data_str = "\n".join(line.strip() for line in data_str.strip().split("\n"))
    df = pd.read_csv(StringIO(data_str), sep="\t", dtype={"group1": str, "group2": int})
    df["interval_start"] = pd.to_datetime(df["interval_start"])
    df["interval_end"] = pd.to_datetime(df["interval_end"])
    return df


one_hour = pd.Timedelta(hours=1)
one_second = pd.Timedelta(seconds=1)


class TestIntervalUnion:
    def test_interval_union_empty_datetime(self):
        assert (
            _interval_union([]) == empty_interval_datetime()
        ), "Failed: Empty list should return an empty interval"

    def test_interval_union_single_datetime_interval(self):
        start_time = pendulum.parse("2020-01-01 00:00:00")
        end_time = pendulum.parse("2020-01-02 00:00:00")
        interv = interval_datetime(start_time, end_time)
        assert (
            _interval_union([interv]) == interv
        ), "Failed: Single interval should return itself"

    def test_interval_union_non_overlapping_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 12:00:00"),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 12:00:00") | interval(
            "2020-01-01 13:00:00", "2020-01-01 23:00:00"
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Non-overlapping intervals not handled correctly"

    def test_interval_union_overlapping_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 15:00:00"),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 23:00:00")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 13:00:00"),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 23:00:00")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00", "2023-01-01 12:00:00"),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2023-01-01 12:00:00")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

    def test_interval_union_adjacent_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 12:59:59"),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 23:00:00")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Adjacent intervals not handled correctly"

    def test_interval_union_mixed_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 12:59:59"),
            interval("2020-01-01 13:00:00", "2020-01-01 15:59:58"),
            interval("2020-01-01 16:00:00", "2020-01-01 23:00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 15:59:58") | interval(
            "2020-01-01 16:00:00", "2020-01-01 23:00:00"
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

    def test_interval_union_timezones(self):
        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:58+00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 15:59:58")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Timezones not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 11:59:59") | interval(
            "2020-01-01 13:00:00", "2020-01-01 15:59:59"
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 13:59:58+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 12:59:58") | interval(
            "2020-01-01 13:00:00", "2020-01-01 15:59:59"
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 13:59:59+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:58+00:00"),
        ]
        expected = interval("2020-01-01 00:00:00", "2020-01-01 15:59:58")
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"


class TestTimestampsToIntervals:
    def test_timestamps_to_intervals_empty(self):
        df = pd.DataFrame(columns=["interval_start", "interval_end"])
        assert (
            timestamps_to_intervals(df) == []
        ), "Failed: Empty DataFrame should return an empty list"

    def test_timestamps_to_intervals_single_row(self):
        df = pd.DataFrame(
            {"interval_start": [1000000000], "interval_end": [2000000000]}
        )
        expected = [interval_int(1, 2)]
        assert (
            timestamps_to_intervals(df) == expected
        ), "Failed: Single row DataFrame not handled correctly"

    def test_timestamps_to_intervals_multiple_rows(self):
        df = pd.DataFrame(
            {
                "interval_start": [1000000000, 3000000000],
                "interval_end": [2000000000, 4000000000],
            }
        )
        expected = [interval_int(1, 2), interval_int(3, 4)]
        assert (
            timestamps_to_intervals(df) == expected
        ), "Failed: Multiple rows DataFrame not handled correctly"

    def test_timestamps_to_intervals_invalid_structure(self):
        df = pd.DataFrame({"start": [1000000000], "end": [2000000000]})
        with pytest.raises(KeyError):
            timestamps_to_intervals(
                df
            ), "Failed: DataFrame with invalid structure should raise KeyError"

    def test_timestamps_to_intervals(self):
        data = {
            "interval_start": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "interval_end": pd.to_datetime(["2023-01-02", "2023-01-03"]),
        }
        df = pd.DataFrame(data)

        result = timestamps_to_intervals(df)

        expected = [
            interval_int(1672531200, 1672617600),
            interval_int(1672617600, 1672704000),
        ]

        assert result == expected


class TestToIntervals:
    def test_to_intervals_empty_dataframe(self):
        df = pd.DataFrame(columns=["interval_start", "interval_end"])
        result = to_intervals(df)
        assert result == [], "Failed: Empty DataFrame should return an empty list"

    def test_to_intervals_single_row(self):
        df = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(["2020-01-01T00:00:00"], utc=True),
                "interval_end": pd.to_datetime(["2020-01-02T00:00:00"], utc=True),
            }
        )
        expected = [interval("2020-01-01T00:00:00", "2020-01-02T00:00:00")]
        result = to_intervals(df)
        assert result == expected, "Failed: Single row DataFrame not handled correctly"

    def test_to_intervals_multiple_rows(self):
        df = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(
                    ["2020-01-01T00:00:00", "2020-01-03T00:00:00"], utc=True
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-02T00:00:00", "2020-01-04T00:00:00"], utc=True
                ),
            }
        )
        expected = [
            interval("2020-01-01T00:00:00", "2020-01-02T00:00:00"),
            interval("2020-01-03T00:00:00", "2020-01-04T00:00:00"),
        ]
        result = to_intervals(df)
        assert (
            result == expected
        ), "Failed: Multiple rows DataFrame not handled correctly"

    def test_to_intervals_invalid_structure(self):
        df = pd.DataFrame(
            {"start": ["2020-01-01T00:00:00"], "end": ["2020-01-02T00:00:00"]}
        )
        with pytest.raises(KeyError):
            to_intervals(
                df
            ), "Failed: DataFrame with invalid structure should raise KeyError"


class TestResultToDf:
    def test_result_to_df_empty(self):
        assert _result_to_df(
            {}, ["group"]
        ).empty, "Failed: Empty result should return empty DataFrame"

    def test_result_to_df_single_group(self):
        result = {("group1",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")]}
        df = _result_to_df(result, ["group"])
        assert (
            len(df) == 1 and df.iloc[0]["group"] == "group1"
        ), "Failed: Single group not handled correctly"

    def test_result_to_df_multiple_groups(self):
        result = {
            ("group1",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")],
            ("group2",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")],
        }
        df = _result_to_df(result, ["group"])
        assert len(df) == 2 and set(df["group"]) == {
            "group1",
            "group2",
        }, "Failed: Multiple groups not handled correctly"

    def test_result_to_df_timezone_handling(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+01:00", "2020-01-02 00:00:00+00:00")
            ]
        }
        df = _result_to_df(result, ["group"])
        assert (
            df["interval_start"].dt.tz.offset == 3600
            and df["interval_end"].dt.tz.offset == 0
        ), "Failed: Timezones not handled correctly"

    def test_result_to_df_group_keys_as_tuples(self):
        result = {
            ("group1", "subgroup1"): [
                interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")
            ]
        }
        df = _result_to_df(result, ["group", "subgroup"])
        assert all(
            df.columns == ["group", "subgroup", "interval_start", "interval_end"]
        ), "Failed: Group keys as tuples not handled correctly"

    def test_result_to_df_timezones(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+01:00", "2020-01-02 00:00:00+01:00")
            ]
        }
        df = _result_to_df(result, ["group"])
        assert (
            df["interval_start"].dt.tz.offset == 3600
            and df["interval_end"].dt.tz.offset == 3600
        )

    def test_result_to_df_single_key(self):
        # Test with single key and specific timezones
        tz_start = pytz.timezone("UTC")
        tz_end = pytz.timezone("America/New_York")

        result = {
            1: [
                interval_datetime(
                    parse_dt("2021-01-01 00:00:00", tz=tz_start),
                    parse_dt("2021-01-01 02:00:00", tz=tz_end),
                ),
                interval_datetime(
                    parse_dt("2021-01-02 00:00:00", tz=tz_start),
                    parse_dt("2021-01-02 02:00:00", tz=tz_end),
                ),
            ]
        }
        by = ["person_id"]

        expected_data = {
            "person_id": [1, 1],
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

        result_df = _result_to_df(result, by)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_result_to_df_multiple_keys(self):
        # Test with multiple keys and None for timezones

        tz_start = pytz.timezone("UTC")
        tz_end = pytz.timezone("America/New_York")

        result = {
            (1, "A"): [
                interval_datetime(
                    parse_dt("2021-01-01 00:00:00", tz=tz_start),
                    parse_dt("2021-01-01 02:00:00", tz=tz_end),
                )
            ],
            (2, "B"): [
                interval_datetime(
                    parse_dt("2021-01-02 00:00:00", tz=tz_start),
                    parse_dt("2021-01-02 02:00:00", tz=tz_end),
                )
            ],
        }
        by = ["person_id", "concept_id"]

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

        result_df = _result_to_df(result, by)

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestInvertIntervals:
    @pytest.fixture
    def observation_window(self):
        return TimeRange(start="2023-01-01 01:00:00Z", end="2023-01-03 16:00:00Z")

    def test_invert_intervals_empty_dataframe(self, observation_window):
        df = pd.DataFrame(columns=["interval_start", "interval_end"])
        result = invert_intervals(df, [], observation_window)
        # Expecting the entire observation window since no intervals to invert
        expected_result = pd.DataFrame(
            columns=["interval_start", "interval_end"]
        )  # empty dataframe
        pd.testing.assert_frame_equal(result, expected_result)

    def test_invert_intervals_single_row(self, observation_window):
        # exactly the same as observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
            }
        )
        expected_result = _result_to_df({(1,): []}, ["person_id"])

        # empty by
        with pytest.raises(ValueError):
            invert_intervals(df, [], observation_window)

        result = invert_intervals(df, ["person_id"], observation_window)
        pd.testing.assert_frame_equal(result, expected_result)

        # longer than observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start - one_hour],
                "interval_end": [observation_window.end + one_hour],
            }
        )
        expected_result = pd.DataFrame(
            columns=["person_id", "interval_start", "interval_end"]
        )
        result = invert_intervals(df, ["person_id"], observation_window)
        pd.testing.assert_frame_equal(result, expected_result)

        # starting at observation window start but ending before end
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end - one_hour],
            }
        )
        expected_result = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.end - one_hour + one_second],
                "interval_end": [observation_window.end],
            }
        )
        result = invert_intervals(df, ["person_id"], observation_window)
        pd.testing.assert_frame_equal(result, expected_result)

        # ending at observation window end but starting after start
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end],
            }
        )
        expected_result = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.start + one_hour - one_second],
            }
        )
        result = invert_intervals(df, ["person_id"], observation_window)
        pd.testing.assert_frame_equal(result, expected_result)

        # in between
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end - one_hour],
            }
        )
        expected_result = pd.DataFrame(
            {
                "person_id": [1, 1],
                "interval_start": [
                    observation_window.start,
                    observation_window.end - one_hour + one_second,
                ],
                "interval_end": [
                    observation_window.start + one_hour - one_second,
                    observation_window.end,
                ],
            }
        )

        result = invert_intervals(df, ["person_id"], observation_window)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_invert_multiple_persons(self):
        by = ["person_id", "concept_id"]
        observation_window = TimeRange(
            name="observation", start="2023-01-01 00:00:00Z", end="2023-01-02 18:00:00Z"
        )

        data = {
            "person_id": [1, 2, 1, 2],
            "concept_id": ["A", "A", "A", "A"],
            "interval_start": pd.to_datetime(
                [
                    "2023-01-01T00:00:00",
                    "2023-01-01T12:00:00",
                    "2023-01-01T06:00:00",
                    "2023-01-02T12:00:00",
                ],
                utc=True,
            ),
            "interval_end": pd.to_datetime(
                [
                    "2023-01-01T12:00:00",
                    "2023-01-02T06:00:00",
                    "2023-01-01T18:00:00",
                    "2023-01-03T12:00:00",
                ],
                utc=True,
            ),
        }
        df = pd.DataFrame(data)

        result = invert_intervals(df, by, observation_window)

        expected_data = {
            "person_id": [1, 2, 2],
            "concept_id": ["A", "A", "A"],
            "interval_start": pd.to_datetime(
                ["2023-01-01T18:00:01", "2023-01-01T0:00:00", "2023-01-02T06:00:01"],
                utc=True,
            ),
            "interval_end": pd.to_datetime(
                [
                    "2023-01-02T18:00:00",
                    "2023-01-01T11:59:59",
                    "2023-01-02T11:59:59",
                ],
                utc=True,
            ),
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the result
        pd.testing.assert_frame_equal(result, expected_df)


class TestFilterCommonItems:
    def test_filter_common_items_empty_dataframe_list(self):
        result = filter_common_items([], ["column1", "column2"])
        assert (
            result == []
        ), "Failed: Empty list of DataFrames should return an empty list"

    def test_filter_common_items_single_dataframe(self):
        # supplying a single data frame should return the same data frame
        df = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        result = filter_common_items([df], ["column1", "column2"])
        pd.testing.assert_frame_equal(result[0], df)

    def test_filter_common_items_multiple_dataframes_common_items(self):
        # compare two dataframes that have 2 common items and 2 different items
        df1 = pd.DataFrame({"column1": [1, 2, 8, 9], "column2": [3, 4, 5, 8]})
        df2 = pd.DataFrame({"column1": [1, 3, 8, 9], "column2": [3, 5, 6, 8]})
        expected_df = pd.DataFrame({"column1": [1, 9], "column2": [3, 8]}, index=[0, 3])
        result = filter_common_items([df1, df2], ["column1", "column2"])
        pd.testing.assert_frame_equal(result[0], expected_df)
        pd.testing.assert_frame_equal(result[1], expected_df)

    def test_filter_common_items_multiple_dataframes_no_common_items(self):
        # compare two dataframes that have no common items
        df1 = pd.DataFrame({"column1": [1], "column2": [3]})
        df2 = pd.DataFrame({"column1": [2], "column2": [4]})
        result = filter_common_items([df1, df2], ["column1", "column2"])
        assert all(
            df.empty for df in result
        ), "Failed: DataFrames with no common items should return empty DataFrames"

    def test_filter_common_items_invalid_columns(self):
        # using an invalid column name should raise a KeyError
        df1 = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        df2 = pd.DataFrame({"column1": [1, 3], "column2": [3, 5]})
        with pytest.raises(KeyError):
            filter_common_items([df1, df2], ["invalid_column"])

    def test_filter_common_items(self):
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


class TestMergeIntervals:
    def test_merge_intervals_empty_dataframe_list(self):
        result = merge_intervals([], ["group"])
        assert (
            result.empty
        ), "Failed: Empty list of DataFrames should return an empty DataFrame"

    def test_merge_intervals_single_dataframe(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A"],
                "interval_start": pd.to_datetime(
                    ["2020-01-01 12:00:00", "2020-01-02 12:00:00"]
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-02 18:00:00", "2020-01-03 18:00:00"]
                ),
            }
        )
        expected_df = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 18:00:00"]),
            }
        )

        result = merge_intervals([df], ["group"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-05 18:00:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-04 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00"]),
            }
        )
        result = merge_intervals([df1, df2], ["group"])
        expected_df = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_non_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
            }
        )
        result = merge_intervals([df1, df2], ["group"])
        expected_df = pd.concat([df1, df2]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_adjacent_intervals(self):
        df1 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:59"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
            }
        )
        result = merge_intervals([df1, df2], ["group"])
        expected_df = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_group_by_multiple_columns(self):
        df1 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-02 12:00:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-02 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00"]),
            }
        )
        result = merge_intervals([df1, df2], ["group1", "group2"])
        expected_df = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_with_timzone(self):
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
                [
                    "2023-01-01T00:00:00Z",
                    "2023-01-02T00:00:00Z",
                    "2023-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "interval_end": pd.to_datetime(
                [
                    "2023-01-01T18:00:00Z",
                    "2023-01-02T12:00:00Z",
                    "2023-01-03T12:00:00Z",
                ],
                utc=True,
            ),
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_merge_intervals_group_by_multiple_columns_complex_data(self):
        data1 = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-01 12:00:00	2023-01-02 12:00:00
        A	1	2023-01-02 05:00:00	2023-01-03 05:00:00
        A	1	2023-01-03 06:00:00	2023-01-03 12:00:00
        A	1	2023-01-03 13:00:00	2023-01-04 12:00:00
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00
        A	2	2023-02-01 12:59:01	2023-02-01 12:59:01
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00
        B	1	2024-01-01 17:00:00	2024-01-01 18:00:00
        B	1	2024-01-01 19:00:00	2024-01-01 20:00:00
        """
        df1 = df_from_str(data1)

        data2 = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-03 12:00:01	2023-01-03 12:59:59
        B	2	2024-02-01 12:00:00	2024-02-01 13:00:00
        B	2	2024-02-01 13:00:00	2024-02-01 14:00:00
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:00
        B	1	2024-01-01 18:00:00	2024-01-01 19:00:00
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00
        """
        df2 = df_from_str(data2)

        data3 = """
        group1	group2	interval_start	interval_end
        A	2	2023-02-01 06:00:00	2023-02-01 06:00:00
        A	2	2023-02-01 06:00:02	2023-02-01 12:58:58
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:01
        """
        df3 = df_from_str(data3)

        expected_data = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-01 12:00:00	2023-01-03 05:00:00
        A	1	2023-01-03 06:00:00	2023-01-04 12:00:00
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:01
        A	2	2023-02-01 06:00:00	2023-02-01 06:00:00
        A	2	2023-02-01 06:00:02	2023-02-01 12:58:58
        B	1	2024-01-01 17:00:00	2024-01-01 20:00:00
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:01
        B	2	2024-02-01 12:00:00	2024-02-01 14:00:00
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00
        """
        expected_df = (
            df_from_str(expected_data)
            .sort_values(by=["group1", "group2", "interval_start", "interval_end"])
            .reset_index(drop=True)
        )

        result = merge_intervals([df1, df2, df3], ["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)


class TestIntersectIntervals:
    def test_intersect_intervals_empty_dataframe_list(self):
        result = intersect_intervals([], ["group"])
        assert (
            result.empty
        ), "Failed: Empty list of DataFrames should return an empty DataFrame"

    def test_intersect_intervals_single_dataframe(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A"],
                "interval_start": pd.to_datetime(
                    ["2020-01-01 08:00:00", "2020-01-02 09:00:00"]
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-01 10:00:00", "2020-01-02 11:00:00"]
                ),
            }
        )
        result = intersect_intervals([df], ["group"])
        expected_df = df.copy()
        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_intersecting_intervals(self):
        df1 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 10:00:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 11:00:00"]),
            }
        )
        result = intersect_intervals([df1, df2], ["group"])
        expected_df = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 10:00:00"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_no_intersecting_intervals(self):
        df1 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 10:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 11:00:00"]),
            }
        )
        result = intersect_intervals([df1, df2], ["group"])
        expected_df = pd.DataFrame(columns=["group", "interval_start", "interval_end"])
        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_group_by_multiple_columns(self):
        df1 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
            }
        )
        df2 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:30:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:30:00"]),
            }
        )
        result = intersect_intervals([df1, df2], ["group1", "group2"])
        expected_df = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:30:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_interval_with_timezone(self):
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

    def test_intersect_intervals_group_by_multiple_columns_complex_data(self):
        data1 = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-01 12:00:00	2023-01-02 12:00:00
        A	1	2023-01-02 05:00:00	2023-01-03 05:00:00
        A	1	2023-01-03 06:00:00	2023-01-03 12:30:00
        A	1	2023-01-03 13:00:00	2023-01-04 12:00:00
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00
        A	2	2023-02-01 12:59:01	2023-02-01 12:59:01
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00
        B	2	2023-02-01 12:00:00	2023-02-01 12:59:00
        B	3	2024-03-01 16:00:00	2024-03-01 20:00:00
        B	4	2024-04-01 12:00:00	2024-04-02 12:00:00
        """
        df1 = df_from_str(data1)

        data2 = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-03 12:00:01	2023-01-03 12:59:59
        A	2	2023-02-01 06:00:00	2023-02-02 12:00:00
        B	2	2024-02-01 13:00:00	2024-02-01 14:00:00
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:00
        B	1	2024-01-01 18:00:00	2024-01-01 19:00:00
        B	1	2024-01-01 13:10:00	2024-01-01 13:20:00
        B	1	2024-01-01 13:30:00	2024-01-01 13:40:00
        B	1	2024-01-01 13:50:00	2024-01-01 14:00:00
        B	2	2023-02-01 12:59:00	2023-02-01 14:00:00
        B	3	2024-03-01 16:00:00	2024-03-01 18:00:00
        B	4	2024-04-01 12:00:00	2024-04-02 12:00:00
        B	5	2024-05-01 16:00:00	2024-05-01 18:00:00
        """
        df2 = df_from_str(data2)

        data3 = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-03 11:00:01	2023-01-03 12:59:59
        A	2	2023-02-01 06:00:02	2023-02-01 12:59:00
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00
        B	1	2024-01-01 00:00:00	2024-01-01 13:25:00
        B	1	2024-01-01 13:45:00	2024-01-01 13:55:00
        B	1	2024-01-01 13:51:00	2024-01-01 13:53:00
        B	2	2023-02-01 12:00:00	2023-02-01 14:00:00
        B	3	2024-03-01 18:00:00	2024-03-01 20:00:00
        B	5	2024-05-01 18:00:00	2024-05-01 20:00:00
        """
        df3 = df_from_str(data3)

        expected_data = """
        group1	group2	interval_start	interval_end
        A	1	2023-01-03 12:00:01	2023-01-03 12:30:00
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00
        B	1	2024-01-01 13:10:00	2024-01-01 13:20:00
        B	1	2024-01-01 13:50:00	2024-01-01 13:55:00
        B	2	2023-02-01 12:59:00	2023-02-01 12:59:00
        B	3	2024-03-01 18:00:00	2024-03-01 18:00:00
        """
        expected_df = df_from_str(expected_data)

        result = intersect_intervals([df1, df2, df3], ["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)
