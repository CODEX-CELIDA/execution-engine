import itertools
from typing import Any

import pandas as pd

from execution_engine.util.interval import IntervalType


class ResultComparator:
    """
    A class representing a combination of recommendation criteria results.

    Helper class to compare recommendation criteria results vs expected in pytest.
    """

    def __init__(self, name: str, df: pd.DataFrame):
        if name not in ["db", "expected"]:
            raise ValueError(f"Invalid name '{name}' for RecommendationCriteriaResult")
        self.name = name

        if df.index.names != ["person_id", "date"]:
            self.df = df.set_index(["person_id", "date"])
        else:
            self.df = df

        # only if there are two levels in the columns
        if isinstance(self.df.columns, pd.MultiIndex):
            # flatten columns
            self.df.columns = ["".join(col) for col in self.df.columns]

        # make NO_DATA and NOT_APPLICABLE equal to False
        with IntervalType.custom_bool_true(
            [IntervalType.POSITIVE, IntervalType.NOT_APPLICABLE]
        ):
            self.df = self.df.astype(bool)

    def __getitem__(self, item: str) -> pd.Series:
        return self.df[item]

    @property
    def plan_names(self) -> list[str]:
        return [col[2:] for col in self.df.columns if col.startswith("i_")]

    def plan_name_column_names(self) -> list[str]:
        cols = [
            "_".join(i) for i in itertools.product(["p", "i", "p_i"], self.plan_names)
        ]
        return cols + ["p", "i", "p_i"]

    def derive_database_result(self, df: pd.DataFrame) -> "ResultComparator":
        df = df.copy()
        df.loc[
            :, [c for c in self.plan_name_column_names() if c not in df.columns]
        ] = False

        return ResultComparator(name="db", df=df)

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, ResultComparator)

        df1, df2 = self.__order_dfs(other)

        def compare_series(name: str) -> bool:
            s1 = df1[name]
            s2 = df2[name]

            # align the series to make sure they have the same index
            s1, s2 = s1.align(s2)
            # replacement for s1.align(s2, fill_value=False) as this raises a warning in pandas 2.2.0
            s1 = s1.astype(bool) & s1.notnull()
            s2 = s2.astype(bool) & s2.notnull()

            return s1.equals(s2)

        return all([compare_series(col) for col in self.plan_name_column_names()])

    def __order_dfs(
        self, other: "ResultComparator"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.name == "expected":
            df1, df2 = self.df, other.df
        elif other.name == "expected":
            df1, df2 = other.df, self.df
        else:
            raise ValueError(
                "Cannot compare two ResultComparator objects that are both not 'expected'"
            )
        return df1, df2

    def comparison_report(self, other: Any) -> list[str]:
        if not isinstance(other, ResultComparator):
            raise ValueError("Can only compare ResultComparator objects")

        if self == other:
            return ["Results match"]

        df1, df2 = self.__order_dfs(other)

        overlapping_cols = self.plan_name_column_names()

        # Find the "other" columns in df1
        other_cols = list(set(df1.columns) - set(overlapping_cols))

        reports = ["Results do not match"]

        # Loop over each person_id
        for person_id in df1.index.get_level_values("person_id").unique():
            df1_subset = df1.loc[person_id]

            if person_id in df2.index.get_level_values("person_id"):
                df2_subset = df2.loc[person_id]
            else:
                df2_subset = pd.DataFrame(
                    index=df1_subset.index, columns=df1_subset.columns, data=False
                )

            # Splitting and sorting the conditions
            true_cols = sorted(
                col for col, val in df1_subset[other_cols].any().items() if val
            )
            false_cols = sorted(
                col for col, val in df1_subset[other_cols].any().items() if not val
            )

            # Constructing the logical expressions
            true_expression = " & ".join(true_cols)
            false_expression = " & ".join(f"~{col}" for col in false_cols)

            mismatch_reported = False
            # Loop over each overlapping column
            for col in overlapping_cols:
                # Find dates where the column doesn't match
                s1 = df1_subset[col]
                s2 = df2_subset[col]
                s1, s2 = s1.align(s2, fill_value=False)
                mismatches = s1[s1 != s2]

                # If any mismatches exist, add a report for this column
                if not mismatches.empty:
                    if not mismatch_reported:
                        person_id_str = f"person_id '{person_id}'"
                        reports.append(f"{person_id_str} - TRUE:  {true_expression}")
                        reports.append(
                            f"{' ' * len(person_id_str)} - FALSE: {false_expression}"
                        )
                        mismatch_reported = True

                    mismatch_reports = []
                    for date, value in mismatches.items():
                        expected_value = value
                        actual_value = s2.loc[date]
                        mismatch_reports.append(
                            f"{date.date()} (expected: {expected_value}, actual: {actual_value})"
                        )

                    reports.append(
                        f"Column '{col}' does not match on dates: {', '.join(mismatch_reports)}"
                    )

        return reports
