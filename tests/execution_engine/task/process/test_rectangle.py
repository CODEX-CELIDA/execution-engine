import random
from datetime import time

import pandas as pd
import pendulum
import pytest
import pytz

from execution_engine.task.process import (
    Interval,
    IntervalWithCount,
    get_processing_module,
)
from execution_engine.util.interval import IntervalType as T
from execution_engine.util.types import PersonIntervals, TimeRange
from tests.functions import df_from_str
from tests.functions import intervals_to_df as intervals_to_df_original
from tests.functions import parse_dt

one_hour = pd.Timedelta(hours=1)
one_second = pd.Timedelta(seconds=1)


@pytest.fixture(params=["cython", "python"], scope="session")
def process_module(request):
    module = get_processing_module("rectangle", version=request.param)
    assert module._impl.MODULE_IMPLEMENTATION == request.param
    return module


def interval(start: str, end: str, type_=T.POSITIVE) -> Interval:
    return Interval(
        pendulum.parse(start).timestamp(), pendulum.parse(end).timestamp(), type_
    )


def df_to_interval_tuple(df: pd.DataFrame) -> Interval:
    """
    Converts the DataFrame to intervals.

    :param df: A DataFrame with columns "interval_start" and "interval_end".
    :return: A list of intervals.
    """

    return [
        Interval(start.timestamp(), end.timestamp(), type_)
        for start, end, type_ in zip(
            df["interval_start"], df["interval_end"], df["interval_type"]
        )
    ]


def df_to_person_interval_tuple(df: pd.DataFrame, by=["person_id"]) -> PersonIntervals:
    return {key: df_to_interval_tuple(group_df) for key, group_df in df.groupby(by=by)}


class ProcessTest:
    @pytest.fixture(autouse=True)
    def setup_method(self, process_module):
        self.process = process_module

    def intervals_to_df(self, result, by):
        return intervals_to_df_original(result, by, self.process.normalize_interval)


class TestToIntervals(ProcessTest):
    def test_to_intervals_empty_dataframe(self):
        df = pd.DataFrame(columns=["interval_start", "interval_end", "interval_type"])
        result = df_to_interval_tuple(df)
        assert result == [], "Failed: Empty DataFrame should return an empty list"

    def test_to_intervals_single_row(self):
        df = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(
                    ["2020-01-01T00:00:00+00:00"], utc=True
                ),
                "interval_end": pd.to_datetime(["2020-01-02T00:00:00+00:00"], utc=True),
                "interval_type": T.POSITIVE,
            }
        )
        expected = [interval("2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00")]
        result = df_to_interval_tuple(df)
        assert result == expected, "Failed: Single row DataFrame not handled correctly"

    def test_to_intervals_multiple_rows(self):
        df = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(
                    ["2020-01-01T00:00:00+00:00", "2020-01-03T00:00:00+00:00"], utc=True
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-02T00:00:00+00:00", "2020-01-04T00:00:00+00:00"], utc=True
                ),
                "interval_type": [T.POSITIVE, T.POSITIVE],
            }
        )
        expected = [
            interval(
                "2020-01-01T00:00:00+00:00",
                "2020-01-02T00:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-03T00:00:00+00:00",
                "2020-01-04T00:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        result = df_to_interval_tuple(df)
        assert (
            result == expected
        ), "Failed: Multiple rows DataFrame not handled correctly"

    def test_to_intervals_invalid_structure(self):
        df = pd.DataFrame(
            {
                "start": ["2020-01-01T00:00:00+00:00"],
                "end": ["2020-01-02T00:00:00+00:00"],
            }
        )
        with pytest.raises(KeyError):
            df_to_interval_tuple(
                df
            ), "Failed: DataFrame with invalid structure should raise KeyError"


class TestResultToDf(ProcessTest):
    def test_result_to_df_empty(self):
        assert self.intervals_to_df(
            {}, ["person_id"]
        ).empty, "Failed: Empty result should return empty DataFrame"

    def test_result_to_df_single_group(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00")
            ]
        }
        df = self.intervals_to_df(result, ["person_id"])
        assert (
            len(df) == 1 and df.iloc[0]["person_id"] == "group1"
        ), "Failed: Single group not handled correctly"

    def test_result_to_df_multiple_groups(self):
        result = {
            ("group1",): [
                interval("2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00")
            ],
            ("group2",): [
                interval("2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00")
            ],
        }
        df = self.intervals_to_df(result, ["person_id"])
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
        df = self.intervals_to_df(result, ["person_id"])
        assert (
            df["interval_start"].dt.tz == pytz.utc
            and df["interval_end"].dt.tz == pytz.utc
        ), "Failed: Timezones not handled correctly"

    def test_result_to_df_group_keys_as_tuples(self):
        result = {
            ("group1", "subgroup1"): [
                interval("2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00")
            ]
        }
        df = self.intervals_to_df(result, ["person_id", "subgroup"])
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
        df = self.intervals_to_df(result, ["person_id"])
        assert (
            df["interval_start"].dt.tz == pytz.utc
            and df["interval_end"].dt.tz == pytz.utc
        )

    def test_result_to_df_single_key(self):
        # Test with single key and specific timezones
        tz_start = pytz.timezone("UTC")
        tz_end = pytz.timezone("America/New_York")

        result = {
            1: [
                Interval(
                    lower=parse_dt("2021-01-01 00:00:00", tz=tz_start).timestamp(),
                    upper=parse_dt("2021-01-01 02:00:00", tz=tz_end).timestamp(),
                    type=T.POSITIVE,
                ),
                Interval(
                    lower=parse_dt("2021-01-02 00:00:00", tz=tz_start).timestamp(),
                    upper=parse_dt("2021-01-02 02:00:00", tz=tz_end).timestamp(),
                    type=T.POSITIVE,
                ),
            ]
        }
        by = ["person_id"]

        expected_data = {
            "person_id": [1, 1],
            "interval_start": [
                pd.Timestamp("2021-01-01 00:00:00", tz=tz_start).tz_convert("UTC"),
                pd.Timestamp("2021-01-02 00:00:00", tz=tz_start).tz_convert("UTC"),
            ],
            "interval_end": [
                pd.Timestamp("2021-01-01 02:00:00", tz=tz_end).tz_convert("UTC"),
                pd.Timestamp("2021-01-02 02:00:00", tz=tz_end).tz_convert("UTC"),
            ],
            "interval_type": [T.POSITIVE, T.POSITIVE],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = self.intervals_to_df(result, by)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_result_to_df_multiple_keys(self):
        # Test with multiple keys and None for timezones

        tz_start = pytz.timezone("UTC")
        tz_end = pytz.timezone("America/New_York")

        result = {
            (1, "A"): [
                Interval(
                    lower=parse_dt("2021-01-01 00:00:00", tz=tz_start).timestamp(),
                    upper=parse_dt("2021-01-01 02:00:00", tz=tz_end).timestamp(),
                    type=T.POSITIVE,
                )
            ],
            (2, "B"): [
                Interval(
                    lower=parse_dt("2021-01-02 00:00:00", tz=tz_start).timestamp(),
                    upper=parse_dt("2021-01-02 02:00:00", tz=tz_end).timestamp(),
                    type=T.POSITIVE,
                )
            ],
        }
        by = ["person_id", "concept_id"]

        expected_data = {
            "person_id": [1, 2],
            "concept_id": ["A", "B"],
            "interval_start": [
                pd.Timestamp("2021-01-01 00:00:00", tz=tz_start).tz_convert("UTC"),
                pd.Timestamp("2021-01-02 00:00:00", tz=tz_start).tz_convert("UTC"),
            ],
            "interval_end": [
                pd.Timestamp("2021-01-01 02:00:00", tz=tz_end).tz_convert("UTC"),
                pd.Timestamp("2021-01-02 02:00:00", tz=tz_end).tz_convert("UTC"),
            ],
            "interval_type": [T.POSITIVE, T.POSITIVE],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = self.intervals_to_df(result, by)

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestComplementIntervals(ProcessTest):
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_complement_intervals_single_row(
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_complementary_intervals_multiple_persons(
        self, observation_window, reference_df
    ):
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

        by = ["person_id"]
        result = self.process.complementary_intervals(
            df_to_person_interval_tuple(df, by=by),
            df_to_person_interval_tuple(reference_df, by=by),
            observation_window,
            interval_type=T.NO_DATA,
        )
        result = self.intervals_to_df(result, by=by)

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


class TestInvertIntervals(ProcessTest):
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

        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])

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

        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
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

        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
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
        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
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
        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
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

        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
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

        result = self.process.invert_intervals(
            df_to_person_interval_tuple(df),
            df_to_person_interval_tuple(reference_df),
            observation_window,
        )
        result = self.intervals_to_df(result, ["person_id"])
        result = result.sort_values(by=["person_id", "interval_start"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(result, expected_df)


class TestFilterCommonKeys(ProcessTest):
    def test_filter_common_items_empty_list(self):
        result = self.process.filter_dicts_by_common_keys([])
        assert result == [], "Failed: Empty list should return an empty list"

    def test_filter_common_items_single_dict(self):
        d = dict(column1=[1, 2], column2=[3, 4])
        result = self.process.filter_dicts_by_common_keys([d])
        assert result[0] == d

    def test_filter_common_items_multiple_dicts_common_items(self):
        d1 = dict(column1=[1, 2, 8, 9], column2=[3, 4, 5, 8])
        d2 = dict(column1=[1, 3, 8, 9], column2=[3, 5, 6, 8])

        result = self.process.filter_dicts_by_common_keys([d1, d2])
        assert d1 == result[0]
        assert d2 == result[1]

    def test_filter_common_items_multiple_dicts_no_common_items(self):
        # compare two dicts that have no common keys
        d1 = dict(column1=[1], column2=[3])
        d2 = dict(column3=[2], column4=[4])
        result = self.process.filter_dicts_by_common_keys([d1, d2])
        assert result == [dict(), dict()]

    def test_filter_common_items(self):
        # Create sample dicts
        d1 = dict(A=1, B=2, C=3)
        d2 = dict(A=1, B=3, D=3)
        d3 = dict(F=1, B=4, D=3)

        # Call the function
        result = self.process.filter_dicts_by_common_keys([d1, d2, d3])

        assert result[0] == dict(B=2)
        assert result[1] == dict(B=3)
        assert result[2] == dict(B=4)


class TestUnionRect(ProcessTest):
    def test_union_rect_negative_duration(self):
        intervals = [
            Interval(lower=5, upper=3, type=T.POSITIVE),
        ]
        with pytest.raises(ValueError):
            # we don't expect this to work at all
            self.process._impl.union_rects(intervals)

    def test_union_rect(self):
        intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
            Interval(lower=3, upper=4, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
            Interval(lower=3, upper=4, type=T.NEGATIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=3, type=T.POSITIVE),
            Interval(lower=3, upper=4, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=3, type=T.POSITIVE),
            Interval(lower=4, upper=4, type=T.NEGATIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=4, type=T.POSITIVE),
            Interval(lower=3, upper=4, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=4, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=4, type=T.POSITIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=4, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=4, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
            Interval(lower=5, upper=6, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
            Interval(lower=5, upper=6, type=T.NEGATIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
            Interval(lower=1, upper=6, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=3, upper=4, type=T.POSITIVE),
            Interval(lower=5, upper=6, type=T.NEGATIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.POSITIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.POSITIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

        intervals = [
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.NEGATIVE),
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        expected_intervals = [
            Interval(lower=1, upper=2, type=T.POSITIVE),
        ]

        result = self.process._impl.union_rects(intervals)

        assert result == expected_intervals

    def test_union_rect_empty_datetime(self):
        assert (
            self.process._impl.union_rects([]) == []
        ), "Failed: Empty list should return an empty interval"

    def test_union_rect_single_datetime_interval(self):
        interv = interval(
            "2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00", type_=T.POSITIVE
        )
        assert self.process._impl.union_rects([interv]) == [
            interv
        ], "Failed: Single interval should return itself"

    def test_union_rect_non_overlapping_datetime_intervals(self):
        intervals = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 12:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-01 13:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        expected = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 12:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-01 13:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Non-overlapping intervals not handled correctly"

    def test_union_rect_overlapping_datetime_intervals(self):
        intervals = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 15:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-01 13:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        expected = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            )
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 13:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-01 13:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        expected = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            )
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

        intervals = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2023-01-01 12:00:00+00:00",
                type_=T.POSITIVE,
            ),
            interval(
                "2020-01-01 13:00:00+00:00",
                "2020-01-01 23:00:00+00:00",
                type_=T.POSITIVE,
            ),
        ]
        expected = [
            interval(
                "2020-01-01 00:00:00+00:00",
                "2023-01-01 12:00:00+00:00",
                type_=T.POSITIVE,
            )
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Overlapping intervals not handled correctly"

    def test_union_rect_adjacent_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 23:00:00+00:00"),
        ]
        expected = [interval("2020-01-01 00:00:00+00:00", "2020-01-01 23:00:00+00:00")]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Adjacent intervals not handled correctly"

    def test_union_rect_mixed_datetime_intervals(self):
        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:58+00:00"),
            interval("2020-01-01 16:00:00+00:00", "2020-01-01 23:00:00+00:00"),
        ]
        expected = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 15:59:58+00:00"),
            interval("2020-01-01 16:00:00+00:00", "2020-01-01 23:00:00+00:00"),
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

    def test_union_rect_timezones(self):
        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:58+00:00"),
        ]
        expected = [interval("2020-01-01 00:00:00+00:00", "2020-01-01 15:59:58+00:00")]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Timezones not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:59+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        expected = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 11:59:59+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 13:59:58+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        expected = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 12:59:58+00:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:59+00:00"),
        ]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"

        intervals = [
            interval("2020-01-01 00:00:00+00:00", "2020-01-01 13:59:59+01:00"),
            interval("2020-01-01 13:00:00+00:00", "2020-01-01 15:59:58+00:00"),
        ]
        expected = [interval("2020-01-01 00:00:00+00:00", "2020-01-01 15:59:58+00:00")]
        assert (
            self.process._impl.union_rects(intervals) == expected
        ), "Failed: Mixed intervals not handled correctly"


class TestMergeAdjacentIntervals(ProcessTest):
    def test_empty_list(self):
        """Test that an empty list returns an empty list."""
        assert self.process._impl.merge_adjacent_intervals([]) == []

    def test_single_interval(self):
        """Test that a list with a single interval returns the same list."""
        intervals = [IntervalWithCount(1, 2, "A", 10)]
        assert self.process._impl.merge_adjacent_intervals(intervals) == intervals

    def test_no_merge_needed(self):
        """Test intervals that do not require merging."""
        intervals = [
            IntervalWithCount(1, 2, "A", 10),
            IntervalWithCount(3, 4, "B", 20),
        ]
        assert self.process._impl.merge_adjacent_intervals(intervals) == intervals

    def test_merge_multiple_adjacent_same_type_and_count(self):
        """Test merging multiple adjacent intervals with the same type and count."""
        intervals = [
            IntervalWithCount(1, 2, "A", 10),
            IntervalWithCount(3, 4, "A", 10),
            IntervalWithCount(5, 6, "A", 10),
        ]
        expected = [IntervalWithCount(1, 6, "A", 10)]
        assert self.process._impl.merge_adjacent_intervals(intervals) == expected

    def test_merge_with_different_types_and_counts(self):
        """Test merging only adjacent intervals with the same type and count, ignoring others."""
        intervals = [
            IntervalWithCount(1, 2, "A", 10),
            IntervalWithCount(3, 4, "A", 10),
            IntervalWithCount(5, 6, "B", 5),
            IntervalWithCount(7, 8, "B", 5),
            IntervalWithCount(9, 10, "A", 10),
        ]
        expected = [
            IntervalWithCount(1, 4, "A", 10),
            IntervalWithCount(5, 8, "B", 5),
            IntervalWithCount(9, 10, "A", 10),
        ]
        assert self.process._impl.merge_adjacent_intervals(intervals) == expected

    def test_non_adjacent_intervals(self):
        """Test handling of non-adjacent intervals."""
        intervals = [
            IntervalWithCount(1, 2, "A", 10),
            IntervalWithCount(4, 5, "A", 10),
            IntervalWithCount(7, 8, "A", 10),
        ]
        # No merging should occur since intervals are not adjacent.
        assert self.process._impl.merge_adjacent_intervals(intervals) == intervals


class TestUnionIntervals(ProcessTest):
    def test_union_intervals_empty_dataframe_list(self):
        result = self.process.union_intervals([])
        assert (
            not result
        ), "Failed: Empty list of DataFrames should return an empty DataFrame"

    def test_union_intervals_single_dataframe(self):
        df = pd.DataFrame(
            {
                "person_id": ["A", "A"],
                "interval_start": pd.to_datetime(
                    ["2020-01-01 12:00:00+00:00", "2020-01-02 12:00:00+00:00"]
                ),
                "interval_end": pd.to_datetime(
                    ["2020-01-02 18:00:00+00:00", "2020-01-03 18:00:00+00:00"]
                ),
                "interval_type": [T.POSITIVE, T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 18:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = self.process.union_intervals([df_to_person_interval_tuple(df)])
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-05 18:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-04 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-06 12:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_non_overlapping_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.concat([df1, df2]).reset_index(drop=True)

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_group_by_multiple_columns(self):
        df1 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-02 12:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-02 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "group1": ["A"],
                "group2": ["B"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 12:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["group1", "group2"]),
                df_to_person_interval_tuple(df2, by=["group1", "group2"]),
            ]
        )
        result = self.intervals_to_df(result, ["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_adjacent_intervals(self):
        df1 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-03 13:30:59+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        df2 = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-03 13:31:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )
        expected_df = pd.DataFrame(
            {
                "person_id": ["A"],
                "interval_start": pd.to_datetime(["2020-01-01 12:00:00+00:00"]),
                "interval_end": pd.to_datetime(["2020-01-04 18:00:00+00:00"]),
                "interval_type": [T.POSITIVE],
            }
        )

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_with_timezone(self):
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

        result_df = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id", "concept_id"]),
                df_to_person_interval_tuple(df2, by=["person_id", "concept_id"]),
            ]
        )
        result_df = self.intervals_to_df(result_df, ["person_id", "concept_id"])

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_union_intervals_group_by_multiple_columns_complex_data(self):
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
        A	1	2023-01-04 18:00:00	2023-01-04 20:00:00	POSITIVE
        A	1	2023-01-04 22:00:00	2023-01-05 02:00:00	POSITIVE
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

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["group1", "group2"]),
                df_to_person_interval_tuple(df2, by=["group1", "group2"]),
                df_to_person_interval_tuple(df3, by=["group1", "group2"]),
            ]
        )
        result = self.intervals_to_df(result, ["group1", "group2"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_edge_case(self):
        data1 = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 13:00:01+00:00	2023-03-02 14:00:00+00:00	POSITIVE
        30833	2023-03-02 14:00:01+00:00	2023-03-02 19:00:00+00:00	NEGATIVE
        """
        df1 = df_from_str(data1)

        data2 = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 14:00:01+00:00	2023-03-02 15:00:00+00:00	POSITIVE
        """
        df2 = df_from_str(data2)

        expected_data = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 13:00:01+00:00	2023-03-02 15:00:00+00:00	POSITIVE
        30833	2023-03-02 15:00:01+00:00	2023-03-02 19:00:00+00:00	NEGATIVE
        """
        expected_df = df_from_str(expected_data)

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

        data1 = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 13:00:01+00:00	2023-03-02 19:00:00+00:00	NEGATIVE
        """
        df1 = df_from_str(data1)

        data2 = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 13:00:01+00:00	2023-03-02 14:00:00+00:00	POSITIVE
        30833	2023-03-02 14:00:01+00:00	2023-03-02 19:00:00+00:00	NEGATIVE
        """
        df2 = df_from_str(data2)

        expected_data = """
        person_id	interval_start	interval_end	interval_type
        30833	2023-03-02 13:00:01+00:00	2023-03-02 14:00:00+00:00	POSITIVE
        30833	2023-03-02 14:00:01+00:00	2023-03-02 19:00:00+00:00	NEGATIVE
        """
        expected_df = df_from_str(expected_data)

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)

    def test_union_intervals_edge_case_int(self):
        intervals1 = {
            1: [
                Interval(lower=1, upper=4, type=T.NEGATIVE),
            ]
        }

        intervals2 = {
            1: [
                Interval(lower=1, upper=2, type=T.POSITIVE),
                Interval(lower=3, upper=4, type=T.NEGATIVE),
            ]
        }

        expected_intervals = {
            1: [
                Interval(lower=1, upper=2, type=T.POSITIVE),
                Interval(lower=3, upper=4, type=T.NEGATIVE),
            ]
        }

        result = self.process.union_intervals([intervals1, intervals2])

        assert len(result) == len(expected_intervals)
        assert result == expected_intervals

    def test_union_intervals_no_data_negative_int(self):
        intervals1 = [
            Interval(lower=1, upper=2, type=T.NO_DATA),
            Interval(lower=3, upper=4, type=T.POSITIVE),
            Interval(lower=5, upper=6, type=T.NO_DATA),
        ]

        intervals2 = [
            Interval(lower=1, upper=2, type=T.NO_DATA),
            Interval(lower=3, upper=4, type=T.NEGATIVE),
            Interval(lower=5, upper=6, type=T.NO_DATA),
        ]

        expected_intervals = {
            1: [
                Interval(lower=1, upper=2, type=T.NO_DATA),
                Interval(lower=3, upper=4, type=T.POSITIVE),
                Interval(lower=5, upper=6, type=T.NO_DATA),
            ]
        }

        result = self.process.union_intervals([{1: intervals1}, {1: intervals2}])

        assert len(result) == len(expected_intervals)
        assert result == expected_intervals

    def test_union_intervals_no_data_negative(self):
        data1 = """
        person_id	interval_start	interval_end	interval_type
        30748	2023-02-26 07:00:00+00:00	2023-03-02 12:59:59+00:00	NO_DATA
        30748	2023-03-02 13:00:00+00:00	2023-03-02 14:00:00+00:00	POSITIVE
        30748	2023-03-02 14:00:01+00:00	2023-04-03 23:00:00+00:00	NO_DATA
        """
        df1 = df_from_str(data1)

        data2 = """
        person_id	interval_start	interval_end	interval_type
        30748	2023-02-26 07:00:00+00:00	2023-03-02 12:59:59+00:00	NO_DATA
        30748	2023-03-02 13:00:00+00:00	2023-03-02 14:00:00+00:00	NEGATIVE
        30748	2023-03-02 14:00:01+00:00	2023-04-03 23:00:00+00:00	NO_DATA
        """

        df2 = df_from_str(data2)

        expected_data = """
        person_id	interval_start	interval_end	interval_type
        30748	2023-02-26 07:00:00+00:00	2023-03-02 12:59:59+00:00	NO_DATA
        30748	2023-03-02 13:00:00+00:00	2023-03-02 14:00:00+00:00	POSITIVE
        30748	2023-03-02 14:00:01+00:00	2023-04-03 23:00:00+00:00	NO_DATA
        """
        expected_df = df_from_str(expected_data)

        result = self.process.union_intervals(
            [
                df_to_person_interval_tuple(df1, by=["person_id"]),
                df_to_person_interval_tuple(df2, by=["person_id"]),
            ]
        )
        result = self.intervals_to_df(result, ["person_id"])

        pd.testing.assert_frame_equal(result, expected_df)
