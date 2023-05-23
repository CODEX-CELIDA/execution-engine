import datetime
import itertools
import re
from abc import ABC

import numpy as np
import pandas as pd
import pytest
import sympy
from sqlalchemy import select
from tqdm import tqdm

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.cdm import Person
from execution_engine.omop.db.result import RecommendationResult
from execution_engine.util import TimeRange
from tests._testdata import concepts
from tests._testdata.parameter import criteria_defs
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
    create_observation,
    create_procedure,
    create_visit,
    generate_binary_combinations_dataframe,
)


class RecommendationCriteriaCombination:
    def __init__(self, name, df):
        if name not in ["db", "expected"]:
            raise ValueError(f"Invalid name '{name}' for RecommendationCriteriaResult")
        self.name = name

        if df.index.names != ["person_id", "date"]:
            self.df = df.set_index(["person_id", "date"])
        else:
            self.df = df

    def __getitem__(self, item):
        return self.df[item]

    @property
    def plan_names(self):
        return [col[2:] for col in self.df.columns if col.startswith("i_")]

    def plan_name_column_names(self):
        cols = [
            "_".join(i) for i in itertools.product(["p", "i", "p_i"], self.plan_names)
        ]
        return cols + ["p", "i", "p_i"]

    def derive_database_result(self, df):
        df = df.copy()
        df.loc[
            :, [c for c in self.plan_name_column_names() if c not in df.columns]
        ] = False

        return RecommendationCriteriaCombination(name="db", df=df)

    def __eq__(self, other):
        assert isinstance(other, RecommendationCriteriaCombination)

        df1, df2 = self.__order_dfs(other)

        def compare_series(name):
            s1 = df1[name]
            s2 = df2[name]
            s1, s2 = s1.align(s2, fill_value=False)
            return s1.equals(s2)

        return all([compare_series(col) for col in self.plan_name_column_names()])

    def __order_dfs(self, other):
        if self.name == "expected":
            df1, df2 = self.df, other.df
        elif other.name == "expected":
            df1, df2 = other.df, self.df
        else:
            raise ValueError(
                "Cannot compare two RecommendationCriteriaCombination objects that are both not 'expected'"
            )
        return df1, df2

    def comparison_report(self, other) -> list[str]:
        if not isinstance(other, RecommendationCriteriaCombination):
            raise ValueError(
                "Can only compare RecommendationCriteriaCombination objects"
            )

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

            # Create a logical expression for "other" columns
            logical_expression = " & ".join(
                f"~{col}" if not val else col
                for col, val in df1_subset[other_cols].any().items()
            )

            if logical_expression:
                reports.append(f"person_id '{person_id}': {logical_expression}")

            # Loop over each overlapping column
            for col in overlapping_cols:
                # Find dates where the column doesn't match
                s1 = df1_subset[col]
                s2 = df2_subset[col]
                s1, s2 = s1.align(s2, fill_value=False)
                mismatches = s1[s1 != s2]

                # If any mismatches exist, add a report for this column
                if not mismatches.empty:
                    mismatch_reports = []
                    for date, value in mismatches.items():
                        expected_value = value
                        actual_value = s2.loc[date]
                        mismatch_reports.append(
                            f"{date} (expected: {expected_value}, actual: {actual_value})"
                        )

                    reports.append(
                        f"Column '{col}' does not match on dates: {', '.join(mismatch_reports)}"
                    )

        return reports


def create_index_from_logical_expression(
    df: pd.DataFrame, expression: str
) -> list[bool]:
    # parse the sympy expression
    parsed_expr = sympy.parse_expr(expression)

    # get the symbols in the expression
    symbols_in_expr = list(parsed_expr.free_symbols)
    symbols_in_expr = [str(symbol) for symbol in symbols_in_expr]

    # create a function from the expression
    func = sympy.lambdify(symbols_in_expr, parsed_expr)

    # calculate the result using the DataFrame columns
    index = func(*[df[symbol] for symbol in symbols_in_expr])

    return index


@pytest.mark.recommendation
class TestRecommendationBase(ABC):
    @pytest.fixture
    def visit_datetime(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 07:00:00", end="2023-03-31 22:00:00", name="visit"
        )

    @pytest.fixture
    def observation_window(self, visit_datetime: TimeRange) -> TimeRange:
        return TimeRange(
            start=visit_datetime.start - datetime.timedelta(days=3),
            end=visit_datetime.end + datetime.timedelta(days=3),
            name="observation",
        )

    @pytest.fixture
    def recommendation_url(self) -> str:
        raise NotImplementedError("Must be implemented by subclass")

    @pytest.fixture
    def population_intervention(self) -> dict:
        raise NotImplementedError("Must be implemented by subclass")

    @pytest.fixture
    def population_intervention_groups(self, population_intervention):
        raise NotImplementedError("Must be implemented by subclass")

    @pytest.fixture
    def invalid_combinations(self, population_intervention):
        raise NotImplementedError("Must be implemented by subclass")

    @pytest.fixture
    def person_combinations(
        self,
        population_intervention: dict,
        run_slow_tests: bool,
    ) -> pd.DataFrame:

        df = generate_binary_combinations_dataframe(population_intervention)

        # Remove invalid combinations
        idx_invalid = df["NADROPARIN_HIGH_WEIGHT"] & df["NADROPARIN_LOW_WEIGHT"]
        df = df[~idx_invalid].copy()

        if not run_slow_tests:
            df = pd.concat([df.head(15), df.tail(15)])

        return df

    @pytest.fixture
    def criteria(
        self,
        person_combinations,
        visit_datetime,
        population_intervention,
    ):

        entries = []

        for person_id, row in tqdm(
            person_combinations.iterrows(),
            total=len(person_combinations),
            desc="Generating criteria",
        ):

            for criterion in population_intervention:

                if not row[criterion]:
                    continue

                params = criteria_defs[criterion]

                entry = {
                    "person_id": person_id,
                    "type": params["type"],
                    "concept": criterion,
                    "concept_id": population_intervention[criterion],
                    "static": params["static"],
                }

                if params["type"] == "condition":
                    entry["start_datetime"] = visit_datetime.start
                    entry["end_datetime"] = visit_datetime.end
                elif params["type"] == "observation":
                    entry["start_datetime"] = datetime.datetime(2023, 3, 15, 12, 0, 0)
                elif params["type"] == "drug":
                    entry["start_datetime"] = datetime.datetime(2023, 3, 2, 12, 0, 0)
                    entry["end_datetime"] = datetime.datetime(2023, 3, 3, 12, 0, 0)
                    entry["quantity"] = (
                        params["dosage_threshold"] - 1
                        if "dosage_threshold" in params
                        else params["dosage"]
                    )
                    entry["quantity"] *= 2  # over two days
                # create_measurement(vo, concepts.LAB_APTT, datetime.datetime(2023,3,4, 18), 50, concepts.UNIT_SECOND)
                elif params["type"] == "visit":
                    entry["start_datetime"] = datetime.datetime(2023, 3, 2, 12, 0, 0)
                    entry["end_datetime"] = datetime.datetime(2023, 3, 18, 12, 0, 0)
                elif params["type"] == "measurement":
                    entry["start_datetime"] = datetime.datetime(2023, 3, 2, 12, 0, 0)
                    entry["end_datetime"] = datetime.datetime(2023, 3, 3, 12, 0, 0)
                    entry["value"] = params["threshold"]
                    entry["unit_concept_id"] = params["unit_concept_id"]
                elif params["type"] == "procedure":
                    entry["start_datetime"] = datetime.datetime(2023, 3, 2, 12, 0, 0)
                    entry["end_datetime"] = datetime.datetime(2023, 3, 3, 12, 0, 0)
                else:
                    raise NotImplementedError(
                        f"Unknown criterion type: {params['type']}"
                    )
                entries.append(entry)

            if row.get("NADROPARIN_HIGH_WEIGHT") or row.get("NADROPARIN_LOW_WEIGHT"):
                entry_weight = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "WEIGHT",
                    "concept_id": concepts.WEIGHT,
                    "start_datetime": datetime.datetime.combine(
                        visit_datetime.start.date(), datetime.time()
                    )
                    + datetime.timedelta(days=1),
                    "value": 71 if row["NADROPARIN_HIGH_WEIGHT"] else 69,
                    "unit_concept_id": concepts.UNIT_KG,
                    "static": True,
                }
                entries.append(entry_weight)
            elif row.get("HEPARIN") or row.get("ARGATROBAN"):
                entry_appt = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "APTT",
                    "concept_id": concepts.LAB_APTT,
                    "start_datetime": entry["start_datetime"]
                    + datetime.timedelta(days=1),
                    "value": 51,
                    "unit_concept_id": concepts.UNIT_SECOND,
                    "static": False,
                }
                entries.append(entry_appt)

        return pd.DataFrame(entries)

    @pytest.fixture
    def insert_criteria(self, db_session, criteria, visit_datetime):
        # db_session.execute(
        #    text("SET session_replication_role = 'replica';")
        # )  # Disable foreign key checks

        for person_id, g in tqdm(
            criteria.groupby("person_id"),
            total=criteria["person_id"].nunique(),
            desc="Inserting criteria",
        ):

            p = Person(
                person_id=person_id,
                gender_concept_id=0,
                year_of_birth=1990,
                month_of_birth=1,
                day_of_birth=1,
                race_concept_id=0,
                ethnicity_concept_id=0,
            )
            vo = create_visit(p.person_id, visit_datetime.start, visit_datetime.end)

            person_entries = [p, vo]

            for _, row in g.iterrows():

                if row["type"] == "condition":
                    entry = create_condition(vo, row["concept_id"])
                elif row["type"] == "observation":
                    entry = create_observation(
                        vo,
                        row["concept_id"],
                        observation_datetime=row["start_datetime"],
                    )
                elif row["type"] == "measurement":
                    entry = create_measurement(
                        vo=vo,
                        measurement_concept_id=row["concept_id"],
                        measurement_datetime=row["start_datetime"],
                        value_as_number=row["value"],
                        unit_concept_id=row["unit_concept_id"],
                    )
                elif row["type"] == "drug":
                    entry = create_drug_exposure(
                        vo=vo,
                        drug_concept_id=row["concept_id"],
                        start_datetime=row["start_datetime"],
                        end_datetime=row["end_datetime"],
                        quantity=row["quantity"],
                    )
                # create_measurement(vo, concepts.LAB_APTT, datetime.datetime(2023,3,4, 18), 50, concepts.UNIT_SECOND)
                elif row["type"] == "visit":
                    entry = create_visit(
                        person_id=vo.person_id,
                        visit_concept_id=row["concept_id"],
                        visit_start_datetime=row["start_datetime"],
                        visit_end_datetime=row["end_datetime"],
                    )
                elif row["type"] == "procedure":
                    entry = create_procedure(
                        vo=vo,
                        procedure_concept_id=row["concept_id"],
                        start_datetime=row["start_datetime"],
                        end_datetime=row["end_datetime"],
                    )

                else:
                    raise NotImplementedError(f"Unknown criterion type {row['type']}")

                person_entries.append(entry)

            db_session.add_all(person_entries)
            db_session.commit()

    @pytest.fixture
    def criteria_extended(
        self,
        insert_criteria: dict,
        criteria: pd.DataFrame,
        population_intervention: dict,
        population_intervention_groups: dict,
        visit_datetime: TimeRange,
        observation_window: TimeRange,
    ) -> pd.DataFrame:

        # assert that population_intervention and population_intervention_groups match
        p = set(
            sum(
                [
                    re.findall(r"\b\w+\b", plan["population"])
                    for plan in population_intervention_groups.values()
                ],
                [],
            )
        )
        i = set(
            sum(
                [
                    re.findall(r"\b\w+\b", plan["intervention"])
                    for plan in population_intervention_groups.values()
                ],
                [],
            )
        )
        assert p | i == set(
            population_intervention.keys()
        ), "population_intervention and population_intervention_groups do not match"

        idx_static = criteria["static"]
        # todo: should be observation_window, not visit_datetime (but that doesn't work, needs to be fixed in the test)
        criteria.loc[idx_static, "start_datetime"] = visit_datetime.start
        criteria.loc[idx_static, "end_datetime"] = visit_datetime.end
        df = self.expand_dataframe_by_date(
            criteria[["person_id", "concept", "start_datetime", "end_datetime"]],
            observation_window=observation_window,
        )
        df.loc[
            :, [c for c in population_intervention.keys() if c not in df.columns]
        ] = False

        for group_name, group in population_intervention_groups.items():
            df[f"p_{group_name}"] = create_index_from_logical_expression(
                df, group["population"]
            )
            df[f"i_{group_name}"] = create_index_from_logical_expression(
                df, group["intervention"]
            )
            df[f"p_i_{group_name}"] = df[f"p_{group_name}"] & df[f"i_{group_name}"]

        df["p"] = df[[c for c in df.columns if c.startswith("p_")]].any(axis=1)
        df["i"] = df[[c for c in df.columns if c.startswith("i_")]].any(axis=1)
        df["p_i"] = df[[c for c in df.columns if c.startswith("p_i_")]].any(axis=1)

        return df

    @staticmethod
    def expand_dataframe_by_date(
        df: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Expand a dataframe with one row per person and one column per concept to a dataframe with one row per person and
        per day and one column per concept between `observation_start_date` and `observation_end_date`.
        """
        df = df.copy()

        # Set end_datetime equal to start_datetime if it's NaT
        df["end_datetime"].fillna(df["start_datetime"], inplace=True)

        # Create a new dataframe with one row for each date (ignoring time) between start_datetime and end_datetime
        df_expanded = pd.concat(
            [
                pd.DataFrame(
                    {
                        "person_id": row["person_id"],
                        "date": pd.date_range(
                            row["start_datetime"].date(),
                            row["end_datetime"].date(),
                            freq="D",
                        ),
                        "concept": row["concept"],
                    },
                    columns=["person_id", "date", "concept"],
                )
                for _, row in df.iterrows()
            ],
            ignore_index=True,
        )

        # Pivot the expanded dataframe to have one column for each unique concept
        df_pivot = df_expanded.pivot_table(
            index=["person_id", "date"], columns="concept", aggfunc=len, fill_value=0
        )

        # Reset index and column names
        df_pivot = df_pivot.reset_index()
        df_pivot.columns.name = None
        df_pivot.columns = ["person_id", "date"] + [col for col in df_pivot.columns[2:]]

        # Fill the new concept columns with True where the condition is met
        df_pivot[df_pivot.columns[2:]] = df_pivot[df_pivot.columns[2:]].applymap(
            lambda x: x > 0
        )

        # Create an auxiliary DataFrame with all combinations of person_id and dates between observation_start_date and observation_end_date
        unique_person_ids = df["person_id"].unique()
        date_range = pd.date_range(
            observation_window.start.date(), observation_window.end.date(), freq="D"
        )
        aux_df = pd.DataFrame(
            {
                "person_id": np.repeat(unique_person_ids, len(date_range)),
                "date": np.tile(date_range, len(unique_person_ids)),
            }
        )

        # Merge the auxiliary DataFrame with the pivoted DataFrame
        merged_df = pd.merge(aux_df, df_pivot, on=["person_id", "date"], how="left")

        # Fill missing values with False
        merged_df[merged_df.columns[2:]] = merged_df[merged_df.columns[2:]].fillna(
            False
        )

        return merged_df

    @staticmethod
    def recommendation_test_runner(
        recommendation_url: str,
        observation_window: TimeRange,
        criteria_extended: pd.DataFrame,
    ) -> None:
        pass

        from execution_engine.clients import omopdb
        from execution_engine.execution_engine import ExecutionEngine

        e = ExecutionEngine(verbose=False)

        print(recommendation_url)
        cdd = e.load_recommendation(recommendation_url, force_reload=False)

        e.execute(
            cdd,
            start_datetime=observation_window.start,
            end_datetime=observation_window.end,
        )
        query = select(RecommendationResult).where(
            RecommendationResult.criterion_name.is_(None)
        )
        df_result = omopdb.query(query)
        df_result["valid_date"] = pd.to_datetime(df_result["valid_date"])
        df_result["name"] = df_result["cohort_category"].map(
            {
                CohortCategory.INTERVENTION: "i",
                CohortCategory.POPULATION: "p",
                CohortCategory.POPULATION_INTERVENTION: "p_i",
            }
        )
        df_result["name"] = df_result.apply(
            lambda row: row["name"]
            if row["recommendation_plan_name"] is None
            else f"{row['name']}_{row['recommendation_plan_name']}",
            axis=1,
        )

        df_result = df_result.rename(columns={"valid_date": "date"})
        df_result = df_result.pivot_table(
            columns="name",
            index=["person_id", "date"],
            values="recommendation_results_id",
            aggfunc=len,
            fill_value=0,
        ).astype(bool)

        result_expected = RecommendationCriteriaCombination(
            name="expected", df=criteria_extended
        )
        result_db = result_expected.derive_database_result(df=df_result)

        assert result_db == result_expected
