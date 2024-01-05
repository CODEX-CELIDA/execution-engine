import datetime
from io import StringIO

import pandas as pd
import pendulum
import pytest
import pytz
from pytz.tzinfo import DstTzInfo

from execution_engine.task import process
from execution_engine.task.process import (
    _interval_union,
    _result_to_df,
    df_to_intervals,
    filter_dataframes_by_shared_column_values,
    intersect_intervals,
    union_intervals,
)
from execution_engine.util.interval import DateTimeInterval
from execution_engine.util.interval import IntervalType
from execution_engine.util.interval import IntervalType as T
from execution_engine.util.interval import empty_interval_datetime, interval_datetime
from execution_engine.util.types import TimeRange

# todo: test interval inversion
# todo: test interval and
# todo: test interval or


def interval(start: str, end: str, type_=T.POSITIVE) -> DateTimeInterval:
    return interval_datetime(pendulum.parse(start), pendulum.parse(end), type_=type_)


def parse_dt(s: str, tz: DstTzInfo) -> datetime.datetime:
    return tz.localize(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))


def df_from_str(data_str):
    data_str = "\n".join(line.strip() for line in data_str.strip().split("\n"))
    df = pd.read_csv(StringIO(data_str), sep="\t", dtype={"group1": str, "group2": int})
    df["interval_start"] = pd.to_datetime(df["interval_start"])
    df["interval_end"] = pd.to_datetime(df["interval_end"])
    df["interval_type"] = df["interval_type"].apply(IntervalType)

    return df


one_hour = pd.Timedelta(hours=1)
one_second = pd.Timedelta(seconds=1)


@pytest.fixture
def empty_dataframe():
    return pd.DataFrame(columns=process.df_dtypes.keys())


class TestIntervalUnion:
    def test_interval_union_empty_datetime(self):
        assert (
            _interval_union([]) == empty_interval_datetime()
        ), "Failed: Empty list should return an empty interval"

    def test_interval_union_single_datetime_interval(self):
        start_time = pendulum.parse("2020-01-01 00:00:00")
        end_time = pendulum.parse("2020-01-02 00:00:00")
        interv = interval_datetime(start_time, end_time, type_=T.POSITIVE)
        assert (
            _interval_union([interv]) == interv
        ), "Failed: Single interval should return itself"

    def test_interval_union_non_overlapping_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 12:00:00", type_=T.POSITIVE),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE),
        ]
        expected = interval(
            "2020-01-01 00:00:00", "2020-01-01 12:00:00", type_=T.POSITIVE
        ) | interval("2020-01-01 13:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE)
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Non-overlapping intervals not handled correctly"

    def test_interval_union_overlapping_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 15:00:00", type_=T.POSITIVE),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE),
        ]
        expected = interval(
            "2020-01-01 00:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00", "2020-01-01 13:00:00", type_=T.POSITIVE),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE),
        ]
        expected = interval(
            "2020-01-01 00:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE
        )
        assert (
            _interval_union(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00", "2023-01-01 12:00:00", type_=T.POSITIVE),
            interval("2020-01-01 13:00:00", "2020-01-01 23:00:00", type_=T.POSITIVE),
        ]
        expected = interval(
            "2020-01-01 00:00:00", "2023-01-01 12:00:00", type_=T.POSITIVE
        )
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


class TestToIntervals:
    def test_to_intervals_empty_dataframe(self):
        df = pd.DataFrame(columns=["interval_start", "interval_end", "interval_type"])
        result = df_to_intervals(df)
        assert result == [], "Failed: Empty DataFrame should return an empty list"

    def test_to_intervals_single_row(self):
        df = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(["2020-01-01T00:00:00"], utc=True),
                "interval_end": pd.to_datetime(["2020-01-02T00:00:00"], utc=True),
                "interval_type": T.POSITIVE,
            }
        )
        expected = [interval("2020-01-01T00:00:00", "2020-01-02T00:00:00")]
        result = df_to_intervals(df)
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
                "interval_type": [T.POSITIVE, T.POSITIVE],
            }
        )
        expected = [
            interval("2020-01-01T00:00:00", "2020-01-02T00:00:00", type_=T.POSITIVE),
            interval("2020-01-03T00:00:00", "2020-01-04T00:00:00", type_=T.POSITIVE),
        ]
        result = df_to_intervals(df)
        assert (
            result == expected
        ), "Failed: Multiple rows DataFrame not handled correctly"

    def test_to_intervals_invalid_structure(self):
        df = pd.DataFrame(
            {"start": ["2020-01-01T00:00:00"], "end": ["2020-01-02T00:00:00"]}
        )
        with pytest.raises(KeyError):
            df_to_intervals(
                df
            ), "Failed: DataFrame with invalid structure should raise KeyError"


class TestResultToDf:
    def test_result_to_df_empty(self):
        assert _result_to_df(
            {}, ["person_id"]
        ).empty, "Failed: Empty result should return empty DataFrame"

    def test_result_to_df_single_group(self):
        result = {("group1",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")]}
        df = _result_to_df(result, ["person_id"])
        assert (
            len(df) == 1 and df.iloc[0]["person_id"] == "group1"
        ), "Failed: Single group not handled correctly"

    def test_result_to_df_multiple_groups(self):
        result = {
            ("group1",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")],
            ("group2",): [interval("2020-01-01 00:00:00", "2020-01-02 00:00:00")],
        }
        df = _result_to_df(result, ["person_id"])
        assert len(df) == 2 and set(df["person_id"]) == {
            "group1",
            "group2",
        }, "Failed: Multiple groups not handled correctly"

    def test_result_to_df_timezone_handling(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+01:00", "2020-01-02 00:00:00+00:00")
            ]
        }
        df = _result_to_df(result, ["person_id"])
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
        df = _result_to_df(result, ["person_id", "subgroup"])
        assert all(
            df.columns
            == [
                "person_id",
                "subgroup",
                "interval_start",
                "interval_end",
                "interval_type",
            ]
        ), "Failed: Group keys as tuples not handled correctly"

    def test_result_to_df_timezones(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+01:00", "2020-01-02 00:00:00+01:00")
            ]
        }
        df = _result_to_df(result, ["person_id"])
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
                    type_=T.POSITIVE,
                ),
                interval_datetime(
                    parse_dt("2021-01-02 00:00:00", tz=tz_start),
                    parse_dt("2021-01-02 02:00:00", tz=tz_end),
                    type_=T.POSITIVE,
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
            "interval_type": [T.POSITIVE, T.POSITIVE],
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
                    type_=T.POSITIVE,
                )
            ],
            (2, "B"): [
                interval_datetime(
                    parse_dt("2021-01-02 00:00:00", tz=tz_start),
                    parse_dt("2021-01-02 02:00:00", tz=tz_end),
                    type_=T.POSITIVE,
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
            "interval_type": [T.POSITIVE, T.POSITIVE],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = _result_to_df(result, by)

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestComplementIntervals:
    @pytest.fixture
    def observation_window(self):
        return TimeRange(start="2023-01-01 01:00:00Z", end="2023-01-03 16:00:00Z")

    @pytest.fixture
    def reference_df(self, observation_window):
        return pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )

    def test_complement_intervals_empty_dataframe(
        self, observation_window, empty_dataframe
    ):
        df = empty_dataframe.copy()
        reference_df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )

        expected_result = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.NO_DATA],
            }
        )

        # workaround, see https://github.com/pandas-dev/pandas/issues/56733
        expected_result[["interval_start", "interval_end"]] = expected_result[
            ["interval_start", "interval_end"]
        ].astype("datetime64[us, UTC]")

        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

    def test_invert_intervals_single_row(
        self, observation_window, empty_dataframe, reference_df
    ):
        # exactly the same as observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = empty_dataframe

        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # longer than observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start - one_hour],
                "interval_end": [observation_window.end + one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = empty_dataframe.copy()

        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # starting at observation window start but ending before end
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.end - one_hour + one_second,
                    observation_window.end,
                    T.NO_DATA,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )
        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # ending at observation window end but starting after start
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.start,
                    observation_window.start + one_hour - one_second,
                    T.NO_DATA,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )
        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # in between
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.start,
                    observation_window.start + one_hour - one_second,
                    T.NO_DATA,
                ),
                (
                    1,
                    observation_window.end - one_hour + one_second,
                    observation_window.end,
                    T.NO_DATA,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )
        pd.testing.assert_frame_equal(result, expected_result)

    def test_multiple_persons(self, observation_window, reference_df):
        data = [
            (
                1,
                pd.to_datetime("2023-01-01 00:00:00+0000"),
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                pd.to_datetime("2023-01-02 06:00:00+0000"),
                T.POSITIVE,
            ),
            (
                1,
                pd.to_datetime("2023-01-01 06:00:00+0000"),
                pd.to_datetime("2023-01-01 18:00:00+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-02 12:00:00+0000"),
                pd.to_datetime("2023-01-03 12:00:00+0000"),
                T.POSITIVE,
            ),
        ]

        df = pd.DataFrame(
            data,
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        result = process.complementary_intervals(
            df, reference_df, observation_window, interval_type=T.NO_DATA
        )

        expected_data = [
            (
                1,
                pd.to_datetime("2023-01-01 18:00:01+0000"),
                observation_window.end,
                T.NO_DATA,
            ),
            (
                2,
                observation_window.start,
                pd.to_datetime("2023-01-01 11:59:59+0000"),
                T.NO_DATA,
            ),
            (
                2,
                pd.to_datetime("2023-01-02 06:00:01+0000"),
                pd.to_datetime("2023-01-02 11:59:59+0000"),
                T.NO_DATA,
            ),
            (
                2,
                pd.to_datetime("2023-01-03 12:00:01+0000"),
                observation_window.end,
                T.NO_DATA,
            ),
        ]
        expected_df = pd.DataFrame(
            expected_data,
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        # Assert the result
        pd.testing.assert_frame_equal(result, expected_df)


class TestInvertIntervals:
    @pytest.fixture
    def observation_window(self):
        return TimeRange(start="2023-01-01 01:00:00Z", end="2023-01-03 16:00:00Z")

    @pytest.fixture
    def empty_dataframe(self):
        return pd.DataFrame(columns=process.df_dtypes.keys())

    @pytest.fixture
    def reference_df(self, observation_window):
        return pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )

    def test_invert_intervals_empty_dataframe(
        self, observation_window, empty_dataframe
    ):
        df = empty_dataframe.copy()
        reference_df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )

        # full observation window (as no data is in the df)
        expected_result = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )

        # workaround, see https://github.com/pandas-dev/pandas/issues/56733
        expected_result[["interval_start", "interval_end"]] = expected_result[
            ["interval_start", "interval_end"]
        ].astype("datetime64[us, UTC]")

        result = process.invert_intervals(df, reference_df, observation_window)

        pd.testing.assert_frame_equal(result, expected_result)

    def test_invert_intervals_single_row(
        self, observation_window, empty_dataframe, reference_df
    ):
        # exactly the same as observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = df.assign(interval_type=T.NEGATIVE)

        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # longer than observation window
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start - one_hour],
                "interval_end": [observation_window.end + one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = df.assign(interval_type=T.NEGATIVE)

        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # starting at observation window start but ending before end
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.start,
                    observation_window.end - one_hour,
                    T.NEGATIVE,
                ),
                (
                    1,
                    observation_window.end - one_hour + one_second,
                    observation_window.end,
                    T.POSITIVE,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )
        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # ending at observation window end but starting after start
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.start,
                    observation_window.start + one_hour - one_second,
                    T.POSITIVE,
                ),
                (
                    1,
                    observation_window.start + one_hour,
                    observation_window.end,
                    T.NEGATIVE,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )
        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # in between
        df = pd.DataFrame(
            {
                "person_id": [1],
                "interval_start": [observation_window.start + one_hour],
                "interval_end": [observation_window.end - one_hour],
                "interval_type": [T.POSITIVE],
            }
        )
        expected_result = pd.DataFrame(
            data=[
                (
                    1,
                    observation_window.start,
                    observation_window.start + one_hour - one_second,
                    T.POSITIVE,
                ),
                (
                    1,
                    observation_window.start + one_hour,
                    observation_window.end - one_hour,
                    T.NEGATIVE,
                ),
                (
                    1,
                    observation_window.end - one_hour + one_second,
                    observation_window.end,
                    T.POSITIVE,
                ),
            ],
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

    def test_multiple_persons(self, observation_window, reference_df):
        data = [
            (
                1,
                pd.to_datetime("2023-01-01 00:00:00+0000"),
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                pd.to_datetime("2023-01-02 06:00:00+0000"),
                T.POSITIVE,
            ),
            (
                1,
                pd.to_datetime("2023-01-01 06:00:00+0000"),
                pd.to_datetime("2023-01-01 18:00:00+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-02 12:00:00+0000"),
                pd.to_datetime("2023-01-03 12:00:00+0000"),
                T.POSITIVE,
            ),
        ]
        df = pd.DataFrame(
            data,
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        expected_data = [
            (
                1,
                pd.to_datetime("2023-01-01 00:00:00+0000"),
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                T.NEGATIVE,
            ),
            (
                1,
                pd.to_datetime("2023-01-01 06:00:00+0000"),
                pd.to_datetime("2023-01-01 18:00:00+0000"),
                T.NEGATIVE,
            ),
            (
                1,
                pd.to_datetime("2023-01-01 18:00:01+0000"),
                observation_window.end,
                T.POSITIVE,
            ),
            (
                2,
                observation_window.start,
                pd.to_datetime("2023-01-01 11:59:59+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-01 12:00:00+0000"),
                pd.to_datetime("2023-01-02 06:00:00+0000"),
                T.NEGATIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-02 06:00:01+0000"),
                pd.to_datetime("2023-01-02 11:59:59+0000"),
                T.POSITIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-02 12:00:00+0000"),
                pd.to_datetime("2023-01-03 12:00:00+0000"),
                T.NEGATIVE,
            ),
            (
                2,
                pd.to_datetime("2023-01-03 12:00:01+0000"),
                observation_window.end,
                T.POSITIVE,
            ),
        ]
        expected_df = pd.DataFrame(
            expected_data,
            columns=["person_id", "interval_start", "interval_end", "interval_type"],
        )

        result = process.invert_intervals(df, reference_df, observation_window)
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(result, expected_df)


class TestFilterCommonItems:
    def test_filter_common_items_empty_dataframe_list(self):
        result = filter_dataframes_by_shared_column_values([], ["column1", "column2"])
        assert (
            result == []
        ), "Failed: Empty list of DataFrames should return an empty list"

    def test_filter_common_items_single_dataframe(self):
        # supplying a single data frame should return the same data frame
        df = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        result = filter_dataframes_by_shared_column_values([df], ["column1", "column2"])
        pd.testing.assert_frame_equal(result[0], df)

    def test_filter_common_items_multiple_dataframes_common_items(self):
        # compare two dataframes that have 2 common items and 2 different items
        df1 = pd.DataFrame({"column1": [1, 2, 8, 9], "column2": [3, 4, 5, 8]})
        df2 = pd.DataFrame({"column1": [1, 3, 8, 9], "column2": [3, 5, 6, 8]})
        expected_df = pd.DataFrame({"column1": [1, 9], "column2": [3, 8]})
        result = filter_dataframes_by_shared_column_values(
            [df1, df2], ["column1", "column2"]
        )
        pd.testing.assert_frame_equal(result[0], expected_df)
        pd.testing.assert_frame_equal(result[1], expected_df)

    def test_filter_common_items_multiple_dataframes_no_common_items(self):
        # compare two dataframes that have no common items
        df1 = pd.DataFrame({"column1": [1], "column2": [3]})
        df2 = pd.DataFrame({"column1": [2], "column2": [4]})
        result = filter_dataframes_by_shared_column_values(
            [df1, df2], ["column1", "column2"]
        )
        assert all(
            df.empty for df in result
        ), "Failed: DataFrames with no common items should return empty DataFrames"

    def test_filter_common_items_invalid_columns(self):
        # using an invalid column name should raise a KeyError
        df1 = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        df2 = pd.DataFrame({"column1": [1, 3], "column2": [3, 5]})
        with pytest.raises(KeyError):
            filter_dataframes_by_shared_column_values([df1, df2], ["invalid_column"])

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
        filtered_dfs = filter_dataframes_by_shared_column_values(
            [df1, df2, df3], ["A", "B"]
        )

        # Define expected output
        expected_df1 = pd.DataFrame(
            {
                "A": [3],
                "B": ["z"],
                "C": ["a3"],
            }
        )
        expected_df2 = pd.DataFrame(
            {
                "A": [3],
                "B": ["z"],
                "C": ["b1"],
            }
        )
        expected_df3 = pd.DataFrame(
            {
                "A": [3],
                "B": ["z"],
                "C": ["c2"],
            }
        )

        # Assert
        pd.testing.assert_frame_equal(filtered_dfs[0], expected_df1)
        pd.testing.assert_frame_equal(filtered_dfs[1], expected_df2)
        pd.testing.assert_frame_equal(filtered_dfs[2], expected_df3)


class TestMergeIntervals:
    def test_merge_intervals_empty_dataframe_list(self):
        result = union_intervals([], by=["person_id"])
        assert (
            result.empty
        ), "Failed: Empty list of DataFrames should return an empty DataFrame"

    def test_merge_intervals_single_dataframe(self):
        df = pd.DataFrame(
            {
                "person_id": ["A", "A"],
                "interval_start": pd.to_datetime(
                    ["2020-01-01 12:00:00", "2020-01-02 12:00:00"]
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-02 18:00:00", "2020-01-03 18:00:00"]
                ),
                "interval_type": [T.POSITIVE, T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 18:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = union_intervals([df], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-05 18:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-04 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = union_intervals([df1, df2], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_non_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.concat([df1, df2]).reset_index(drop=True)

        result = union_intervals([df1, df2], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_group_by_multiple_columns(self):
        df1 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-02 12:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-02 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = union_intervals([df1, df2], by=["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_adjacent_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:59"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = union_intervals([df1, df2], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_merge_intervals_with_timzone(self):
        data1 = {
            "person_id": [1, 1],
            "concept_id": ["A", "A"],
            "interval_start": pd.to_datetime(
                ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"], utc=True
            ),
            "interval_end": pd.to_datetime(
                ["2023-01-01T12:00:00Z", "2023-01-02T12:00:00Z"], utc=True
            ),
            "interval_type": [T.POSITIVE, T.POSITIVE],
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
            "interval_type": [T.POSITIVE, T.POSITIVE],
        }
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

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
            "interval_type": [T.POSITIVE, T.POSITIVE, T.POSITIVE],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = union_intervals([df1, df2], by=["person_id", "concept_id"])

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_merge_intervals_group_by_multiple_columns_complex_data(self):
        data1 = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-01 12:00:00	2023-01-02 12:00:00	POSITIVE
        A	1	2023-01-02 05:00:00	2023-01-03 05:00:00	POSITIVE
        A	1	2023-01-03 06:00:00	2023-01-03 12:00:00	POSITIVE
        A	1	2023-01-03 13:00:00	2023-01-04 12:00:00	POSITIVE
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00	POSITIVE
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00	POSITIVE
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00	POSITIVE
        A	2	2023-02-01 12:59:01	2023-02-01 12:59:01	POSITIVE
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00	POSITIVE
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00	POSITIVE
        B	1	2024-01-01 17:00:00	2024-01-01 18:00:00	POSITIVE
        B	1	2024-01-01 19:00:00	2024-01-01 20:00:00	POSITIVE
        """
        df1 = df_from_str(data1)

        data2 = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-03 12:00:01	2023-01-03 12:59:59	POSITIVE
        B	2	2024-02-01 12:00:00	2024-02-01 13:00:00	POSITIVE
        B	2	2024-02-01 13:00:00	2024-02-01 14:00:00	POSITIVE
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:00	POSITIVE
        B	1	2024-01-01 18:00:00	2024-01-01 19:00:00	POSITIVE
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00	POSITIVE
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00	POSITIVE
        """
        df2 = df_from_str(data2)

        data3 = """
        group1	group2	interval_start	interval_end	interval_type
        A	2	2023-02-01 06:00:00	2023-02-01 06:00:00	POSITIVE
        A	2	2023-02-01 06:00:02	2023-02-01 12:58:58	POSITIVE
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00	POSITIVE
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00	POSITIVE
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:01	POSITIVE
        """
        df3 = df_from_str(data3)

        expected_data = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-01 12:00:00	2023-01-03 05:00:00	POSITIVE
        A	1	2023-01-03 06:00:00	2023-01-04 12:00:00	POSITIVE
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00	POSITIVE
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00	POSITIVE
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00	POSITIVE
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:01	POSITIVE
        A	2	2023-02-01 06:00:00	2023-02-01 06:00:00	POSITIVE
        A	2	2023-02-01 06:00:02	2023-02-01 12:58:58	POSITIVE
        B	1	2024-01-01 17:00:00	2024-01-01 20:00:00	POSITIVE
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00	POSITIVE
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00	POSITIVE
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:01	POSITIVE
        B	2	2024-02-01 12:00:00	2024-02-01 14:00:00	POSITIVE
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00	POSITIVE
        """
        expected_df = (
            df_from_str(expected_data)
            .sort_values(by=["group1", "group2", "interval_start", "interval_end"])
            .reset_index(drop=True)
        )

        result = union_intervals([df1, df2, df3], by=["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)


class TestIntersectIntervals:
    def test_intersect_intervals_empty_dataframe_list(self):
        result = intersect_intervals([], by=["person_id"])
        assert (
            result.empty
        ), "Failed: Empty list of DataFrames should return an empty DataFrame"

    def test_intersect_intervals_single_dataframe(self):
        df = pd.DataFrame(
            {
                "person_id": ["A", "A"],
                "interval_start": pd.to_datetime(
                    ["2020-01-01 08:00:00", "2020-01-02 09:00:00"]
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-01 10:00:00", "2020-01-02 11:00:00"]
                ),
                "interval_type": [T.POSITIVE, T.POSITIVE],
            }
        )
        result = intersect_intervals([df], by=["person_id"])
        expected_df = df.copy()
        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_intersecting_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 10:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 11:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 10:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = intersect_intervals([df1, df2], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_no_intersecting_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 10:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 11:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            columns=["person_id", "interval_start", "interval_end", "interval_type"]
        )

        result = intersect_intervals([df1, df2], by=["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_intersect_intervals_group_by_multiple_columns(self):
        df1 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:00:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:30:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:30:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 08:30:00"]),
                "interval_end": pd.to_datetime(["2020-01-01 09:00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = intersect_intervals([df1, df2], by=["group1", "group2"])

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
            "interval_type": [T.POSITIVE, T.POSITIVE],
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
            "interval_type": [T.POSITIVE, T.POSITIVE],
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
            "interval_type": [T.POSITIVE],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_intersect_intervals_group_by_multiple_columns_complex_data(self):
        data1 = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-01 12:00:00	2023-01-02 12:00:00	POSITIVE
        A	1	2023-01-02 05:00:00	2023-01-03 05:00:00	POSITIVE
        A	1	2023-01-03 06:00:00	2023-01-03 12:30:00	POSITIVE
        A	1	2023-01-03 13:00:00	2023-01-04 12:00:00	POSITIVE
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00	POSITIVE
        A	1	2023-01-05 06:00:00	2023-01-05 23:59:00	POSITIVE
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00	POSITIVE
        A	2	2023-02-01 12:59:01	2023-02-01 12:59:01	POSITIVE
        B	1	2024-01-01 13:00:00	2024-01-01 14:00:00	POSITIVE
        B	1	2024-01-01 15:00:00	2024-01-01 16:00:00	POSITIVE
        B	2	2023-02-01 12:00:00	2023-02-01 12:59:00	POSITIVE
        B	3	2024-03-01 16:00:00	2024-03-01 20:00:00	POSITIVE
        B	4	2024-04-01 12:00:00	2024-04-02 12:00:00	POSITIVE
        """
        df1 = df_from_str(data1)

        data2 = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-03 12:00:01	2023-01-03 12:59:59	POSITIVE
        A	2	2023-02-01 06:00:00	2023-02-02 12:00:00	POSITIVE
        B	2	2024-02-01 13:00:00	2024-02-01 14:00:00	POSITIVE
        B	2	2024-02-01 15:00:00	2024-02-01 16:00:00	POSITIVE
        B	1	2024-01-01 18:00:00	2024-01-01 19:00:00	POSITIVE
        B	1	2024-01-01 13:10:00	2024-01-01 13:20:00	POSITIVE
        B	1	2024-01-01 13:30:00	2024-01-01 13:40:00	POSITIVE
        B	1	2024-01-01 13:50:00	2024-01-01 14:00:00	POSITIVE
        B	2	2023-02-01 12:59:00	2023-02-01 14:00:00	POSITIVE
        B	3	2024-03-01 16:00:00	2024-03-01 18:00:00	POSITIVE
        B	4	2024-04-01 12:00:00	2024-04-02 12:00:00	POSITIVE
        B	5	2024-05-01 16:00:00	2024-05-01 18:00:00	POSITIVE
        """
        df2 = df_from_str(data2)

        data3 = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-03 11:00:01	2023-01-03 12:59:59	POSITIVE
        A	2	2023-02-01 06:00:02	2023-02-01 12:59:00	POSITIVE
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00	POSITIVE
        B	3	2023-03-04 22:00:00	2023-03-05 02:00:00	POSITIVE
        B	1	2024-01-01 00:00:00	2024-01-01 13:25:00	POSITIVE
        B	1	2024-01-01 13:45:00	2024-01-01 13:55:00	POSITIVE
        B	1	2024-01-01 13:51:00	2024-01-01 13:53:00	POSITIVE
        B	2	2023-02-01 12:00:00	2023-02-01 14:00:00	POSITIVE
        B	3	2024-03-01 18:00:00	2024-03-01 20:00:00	POSITIVE
        B	5	2024-05-01 18:00:00	2024-05-01 20:00:00	POSITIVE
        """
        df3 = df_from_str(data3)

        expected_data = """
        group1	group2	interval_start	interval_end	interval_type
        A	1	2023-01-03 12:00:01	2023-01-03 12:30:00	POSITIVE
        A	2	2023-02-01 12:59:00	2023-02-01 12:59:00	POSITIVE
        B	1	2024-01-01 13:10:00	2024-01-01 13:20:00	POSITIVE
        B	1	2024-01-01 13:50:00	2024-01-01 13:55:00	POSITIVE
        B	2	2023-02-01 12:59:00	2023-02-01 12:59:00	POSITIVE
        B	3	2024-03-01 18:00:00	2024-03-01 18:00:00	POSITIVE
        """
        expected_df = df_from_str(expected_data)

        result = intersect_intervals([df1, df2, df3], ["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)


class TestIntervalFilling:
    @staticmethod
    def assert_equal(data, expected):
        def to_df(data):
            return pd.DataFrame(
                data,
                columns=[
                    "person_id",
                    "interval_start",
                    "interval_end",
                    "interval_type",
                ],
            )

        df_result = process.forward_fill(to_df(data))
        df_expected = to_df(expected)

        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_single_row(self):
        data = [
            (1, "2023-03-01 08:00:00", "2023-03-01 08:00:00", "POSITIVE"),
        ]

        expected = [
            (1, "2023-03-01 08:00:00", "2023-03-01 08:00:00", "POSITIVE"),
        ]

        self.assert_equal(data, expected)

    def test_empty(self):
        data = []
        expected = []

        self.assert_equal(data, expected)

    def test_single_type_per_person(self):
        data = [
            (1, "2023-03-01 08:00:00", "2023-03-01 08:00:00", "POSITIVE"),
            (1, "2023-03-01 09:00:00", "2023-03-01 09:00:00", "POSITIVE"),
            (1, "2023-03-01 09:10:00", "2023-03-01 10:00:00", "POSITIVE"),
            (2, "2023-03-01 11:00:00", "2023-03-01 11:00:00", "POSITIVE"),
            (2, "2023-03-01 12:00:00", "2023-03-01 13:00:00", "POSITIVE"),
            (2, "2023-03-01 14:00:00", "2023-03-01 15:00:00", "POSITIVE"),
        ]

        expected = [
            (1, "2023-03-01 08:00:00", "2023-03-01 10:00:00", "POSITIVE"),
            (2, "2023-03-01 11:00:00", "2023-03-01 15:00:00", "POSITIVE"),
        ]

        self.assert_equal(data, expected)

    def test_last_row_different(self):
        data = [
            (1, "2023-03-01 08:00:00", "2023-03-01 08:00:00", "POSITIVE"),
            (1, "2023-03-01 09:00:00", "2023-03-01 09:00:00", "POSITIVE"),
            (1, "2023-03-01 09:10:00", "2023-03-01 10:00:00", "NEGATIVE"),
            (2, "2023-03-01 11:00:00", "2023-03-01 11:00:00", "POSITIVE"),
            (2, "2023-03-01 12:00:00", "2023-03-01 13:00:00", "POSITIVE"),
            (2, "2023-03-01 14:00:00", "2023-03-01 15:00:00", "NEGATIVE"),
        ]

        expected = [
            (1, "2023-03-01 08:00:00", "2023-03-01 09:10:00", "POSITIVE"),
            (1, "2023-03-01 09:10:00", "2023-03-01 10:00:00", "NEGATIVE"),
            (2, "2023-03-01 11:00:00", "2023-03-01 14:00:00", "POSITIVE"),
            (2, "2023-03-01 14:00:00", "2023-03-01 15:00:00", "NEGATIVE"),
        ]

        self.assert_equal(data, expected)

    def test_forward_fill(self):
        data = [
            (1, "2023-03-01 08:00:00", "2023-03-01 08:00:00", "POSITIVE"),
            (1, "2023-03-01 09:00:00", "2023-03-01 09:00:00", "POSITIVE"),
            (1, "2023-03-01 09:10:00", "2023-03-01 10:00:00", "POSITIVE"),
            (1, "2023-03-01 11:00:00", "2023-03-01 11:00:00", "NEGATIVE"),
            (1, "2023-03-01 12:00:00", "2023-03-01 13:00:00", "POSITIVE"),
            (1, "2023-03-01 14:00:00", "2023-03-01 15:00:00", "POSITIVE"),
        ]

        expected = [
            (1, "2023-03-01 08:00:00", "2023-03-01 11:00:00", "POSITIVE"),
            (1, "2023-03-01 11:00:00", "2023-03-01 12:00:00", "NEGATIVE"),
            (1, "2023-03-01 12:00:00", "2023-03-01 15:00:00", "POSITIVE"),
        ]
        self.assert_equal(data, expected)

        data = [
            (1, "2021-01-01 08:00:00", "2021-01-01 09:00:00", "POSITIVE"),
            (1, "2021-01-01 09:00:00", "2021-01-01 10:00:00", "POSITIVE"),
            (2, "2021-01-02 10:00:00", "2021-01-02 10:15:00", "NEGATIVE"),
            (2, "2021-01-02 10:30:00", "2021-01-02 11:00:00", "POSITIVE"),
            (2, "2021-01-02 11:30:00", "2021-01-02 12:00:00", "NEGATIVE"),
            (3, "2021-01-03 12:00:00", "2021-01-03 12:30:00", "POSITIVE"),
            (3, "2021-01-03 12:45:00", "2021-01-03 13:00:00", "NEGATIVE"),
        ]

        expected = [
            (1, "2021-01-01 08:00:00", "2021-01-01 10:00:00", "POSITIVE"),
            (2, "2021-01-02 10:00:00", "2021-01-02 10:30:00", "NEGATIVE"),
            (2, "2021-01-02 10:30:00", "2021-01-02 11:30:00", "POSITIVE"),
            (2, "2021-01-02 11:30:00", "2021-01-02 12:00:00", "NEGATIVE"),
            (3, "2021-01-03 12:00:00", "2021-01-03 12:45:00", "POSITIVE"),
            (3, "2021-01-03 12:45:00", "2021-01-03 13:00:00", "NEGATIVE"),
        ]

        self.assert_equal(data, expected)
