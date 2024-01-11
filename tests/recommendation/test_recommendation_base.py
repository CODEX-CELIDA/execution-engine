import datetime
import itertools
import re
from abc import ABC
from functools import reduce

import numpy as np
import pandas as pd
import pendulum
import pytest
import sympy
from numpy import typing as npt
from sqlalchemy import select
from tqdm import tqdm

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.custom import TidalVolumePerIdealBodyWeight
from execution_engine.omop.db.celida.tables import PopulationInterventionPair
from execution_engine.omop.db.celida.views import (
    full_day_coverage,
    partial_day_coverage,
)
from execution_engine.omop.db.omop.tables import Person
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import TimeRange
from tests._testdata import concepts, parameter
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
    create_observation,
    create_procedure,
    create_visit,
    generate_binary_combinations_dataframe,
)

MISSING_DATA_TYPE = {
    "condition": IntervalType.NEGATIVE,
    "observation": IntervalType.NO_DATA,
    "drug": IntervalType.NEGATIVE,
    "visit": IntervalType.NEGATIVE,
    "measurement": IntervalType.NO_DATA,
    "procedure": IntervalType.NEGATIVE,
}


class RecommendationCriteriaCombination:
    def __init__(self, name, df):
        if name not in ["db", "expected"]:
            raise ValueError(f"Invalid name '{name}' for RecommendationCriteriaResult")
        self.name = name

        if df.index.names != ["person_id", "date"]:
            self.df = df.set_index(["person_id", "date"])
        else:
            self.df = df

        # make NO_DATA and NOT_APPLICABLE equal to False
        with IntervalType.custom_bool_true(
            [IntervalType.POSITIVE, IntervalType.NOT_APPLICABLE]
        ):
            self.df = self.df.astype(bool)

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
                        reports.append(f"person_id '{person_id}': {logical_expression}")
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


def elementwise_mask(s1, mask, fill_value=False):
    return s1.combine(mask, lambda x, y: x if y else fill_value)


def elementwise_and(s1, s2):
    return s1.combine(s2, lambda x, y: x & y)


def elementwise_or(s1, s2):
    return s1.combine(s2, lambda x, y: x | y)


def elementwise_not(s1):
    return s1.map(lambda x: ~x)


def combine_dataframe_via_logical_expression(
    df: pd.DataFrame, expression: str
) -> npt.NDArray:
    def eval_expr(expr):
        if isinstance(expr, sympy.Symbol):
            return df[str(expr)]
        elif isinstance(expr, sympy.And):
            return reduce(elementwise_and, map(eval_expr, expr.args))
        elif isinstance(expr, sympy.Or):
            return reduce(elementwise_or, map(eval_expr, expr.args))
        elif isinstance(expr, sympy.Not):
            return elementwise_not(eval_expr(expr.args[0]))
        else:
            raise ValueError(f"Unsupported expression: {expr}")

    parsed_expr = sympy.parse_expr(expression)

    return eval_expr(parsed_expr)


@pytest.mark.recommendation
class TestRecommendationBase(ABC):
    @pytest.fixture
    def visit_datetime(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 07:00:00+01:00",
            end="2023-03-31 22:00:00+01:00",
            name="visit",
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
    def invalid_combinations(self, population_intervention) -> str:
        return ""

    @pytest.fixture
    def person_combinations(
        self, unique_criteria: set[str], run_slow_tests: bool, invalid_combinations: str
    ) -> pd.DataFrame:
        df = generate_binary_combinations_dataframe(list(unique_criteria))

        # Remove invalid combinations
        if invalid_combinations:
            idx_invalid = combine_dataframe_via_logical_expression(
                df, invalid_combinations
            )
            df = df[~idx_invalid].copy()

        if not run_slow_tests:
            df = pd.concat([df.head(100), df.tail(100)]).drop_duplicates()

        return df

    @pytest.fixture
    def criteria(
        self,
        person_combinations: pd.DataFrame,
        visit_datetime: TimeRange,
        population_intervention: dict,
    ):
        entries = []

        for person_id, row in tqdm(
            person_combinations.iterrows(),
            total=len(person_combinations),
            desc="Generating criteria",
        ):
            for criterion_name, comparator in self.extract_criteria(
                population_intervention
            ):
                criterion: parameter.CriterionDefinition = getattr(
                    parameter, criterion_name
                )

                if not row[criterion_name]:
                    continue

                if criterion.datetime_offset and not criterion.type == "measurement":
                    raise NotImplementedError(
                        "datetime_offset is only implemented for measurements"
                    )
                time_offsets = criterion.datetime_offset or datetime.timedelta()
                if not isinstance(time_offsets, list):
                    time_offsets = [time_offsets]

                add = {">": 1, "<": -1, "=": 0, "": 0}[comparator]

                entry = {
                    "person_id": person_id,
                    "type": criterion.type,
                    "concept": criterion.name,
                    "concept_id": criterion.concept_id,
                    "static": criterion.static,
                }

                if criterion.type == "condition":
                    entry["start_datetime"] = visit_datetime.start
                    entry["end_datetime"] = visit_datetime.end
                elif criterion.type == "observation":
                    entry["start_datetime"] = pendulum.parse(
                        "2023-03-15 12:00:00+01:00"
                    )
                elif criterion.type == "drug":
                    entry["start_datetime"] = pendulum.parse(
                        "2023-03-02 12:00:00+01:00"
                    )
                    entry["end_datetime"] = pendulum.parse("2023-03-03 12:00:00+01:00")
                    entry["quantity"] = (  # type: ignore
                        criterion.dosage_threshold
                        if criterion.dosage_threshold is not None
                        else criterion.dosage
                    ) + add
                    entry["quantity"] *= 2  # over two days

                    assert criterion.doses_per_day is not None
                    if criterion.doses_per_day > 1:  # add more doses
                        entry["quantity"] /= criterion.doses_per_day
                        entries += [
                            entry.copy() for _ in range(criterion.doses_per_day - 1)
                        ]
                # create_measurement(vo, concepts.LAB_APTT, datetime.datetime(2023,3,4, 18), 50, concepts.UNIT_SECOND)
                elif criterion.type == "visit":
                    entry["start_datetime"] = pendulum.parse(
                        "2023-03-02 12:00:00+01:00"
                    )
                    entry["end_datetime"] = pendulum.parse("2023-03-18 12:00:00+01:00")
                elif criterion.type == "measurement":
                    entry["start_datetime"] = pendulum.parse(
                        "2023-03-02 12:00:00+01:00"
                    )
                    assert criterion.threshold is not None
                    entry["value"] = criterion.threshold + add
                    entry["unit_concept_id"] = criterion.unit_concept_id
                elif criterion.type == "procedure":
                    entry["start_datetime"] = pendulum.parse(
                        "2023-03-02 12:00:00+01:00"
                    )
                    if criterion.duration_threshold_hours is not None:
                        entry["end_datetime"] = entry[
                            "start_datetime"
                        ] + datetime.timedelta(
                            hours=criterion.duration_threshold_hours + add
                        )
                    else:
                        entry["end_datetime"] = pendulum.parse(
                            "2023-03-03 12:00:00+01:00"
                        )
                else:
                    raise NotImplementedError(
                        f"Unknown criterion type: {criterion.type}"
                    )

                if time_offsets:
                    for time_offset in time_offsets:
                        current_entry = entry.copy()

                        current_entry["start_datetime"] += time_offset
                        if "end_datetime" in entry:
                            current_entry["end_datetime"] += time_offset

                        entries.append(current_entry)
                else:
                    entries.append(entry)

            if row.get("NADROPARIN_HIGH_WEIGHT") or row.get("NADROPARIN_LOW_WEIGHT"):
                entry_weight = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "WEIGHT",
                    "concept_id": concepts.BODY_WEIGHT,
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
            elif row.get("TIDAL_VOLUME"):
                # need to add height to calculate ideal body weight and then tidal volume per kg
                entry_weight = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "HEIGHT",
                    "concept_id": concepts.BODY_HEIGHT,
                    "start_datetime": entry["start_datetime"]
                    - datetime.timedelta(days=1),
                    "value": TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                        "female", 70
                    ),
                    "unit_concept_id": concepts.UNIT_CM,
                    "static": True,
                }
                entries.append(entry_weight)

        return pd.DataFrame(entries)

    @pytest.fixture
    def insert_criteria(self, db_session, criteria, visit_datetime):
        for person_id, g in tqdm(
            criteria.groupby("person_id"),
            total=criteria["person_id"].nunique(),
            desc="Inserting criteria",
        ):
            p = Person(
                person_id=person_id,
                gender_concept_id=concepts.GENDER_FEMALE,
                year_of_birth=1990,
                month_of_birth=1,
                day_of_birth=1,
                race_concept_id=0,
                ethnicity_concept_id=0,
            )
            vo = create_visit(
                person_id=p.person_id,
                visit_start_datetime=visit_datetime.start,
                visit_end_datetime=visit_datetime.end,
                visit_concept_id=concepts.INPATIENT_VISIT,
            )

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

    @staticmethod
    def extract_criteria(population_intervention) -> list[tuple[str, str]]:
        criteria: list[tuple[str, str]] = sum(
            [
                re.findall(
                    r"(\b\w+\b)([<=>]?)",
                    plan["population"] + " " + plan["intervention"],
                )
                for plan in population_intervention.values()
            ],
            [],
        )

        unique_criteria = list(dict.fromkeys(criteria))  # order preserving

        return unique_criteria

    @pytest.fixture
    def unique_criteria(self, population_intervention) -> list[str]:
        names = [c[0] for c in self.extract_criteria(population_intervention)]
        return list(dict.fromkeys(names))  # order preserving

    @pytest.fixture
    def criteria_extended(
        self,
        insert_criteria: dict,
        criteria: pd.DataFrame,
        unique_criteria: set[tuple[str, str]],
        population_intervention: dict[str, dict],
        visit_datetime: TimeRange,
        observation_window: TimeRange,
    ) -> pd.DataFrame:
        def remove_comparators(s):
            return s.translate(str.maketrans("", "", "<>="))

        idx_static = criteria["static"]
        criteria.loc[idx_static, "start_datetime"] = observation_window.start
        criteria.loc[idx_static, "end_datetime"] = observation_window.end

        df = self.expand_dataframe_by_date(
            criteria[
                ["person_id", "concept", "start_datetime", "end_datetime", "type"]
            ],
            observation_window=observation_window,
        )

        # the base criterion is the visit, all other criteria are AND-combined with the base criterion
        df_base = self.expand_dataframe_by_date(
            pd.DataFrame(
                {
                    "person_id": criteria["person_id"].unique(),
                    "start_datetime": visit_datetime.start,
                    "end_datetime": visit_datetime.end,
                    "concept": "BASE",
                    "type": "visit",
                }
            ),
            observation_window,
        )

        for c in unique_criteria:
            if c not in df.columns:
                df[c] = MISSING_DATA_TYPE[getattr(parameter, "COVID19").type]

        for group_name, group in population_intervention.items():
            df[f"p_{group_name}"] = combine_dataframe_via_logical_expression(
                df, remove_comparators(group["population"])
            )
            df[f"i_{group_name}"] = combine_dataframe_via_logical_expression(
                df, remove_comparators(group["intervention"])
            )
            # P&I is handled differently than the usual intersection priority order of IntervalType, which is
            # NEGATIVE > POSITIVE > NO_DATA > NOT_APPLICABLE, => POSITIVE & NO_DATA = POSITIVE
            # Here, we need: NEGATIVE > NO_DATA > POSITIVE > NOT_APPLICABLE, so that POSITIVE & NO_DATA = NO_DATA
            with IntervalType.custom_intersection_priority_order(
                order=[
                    IntervalType.NEGATIVE,
                    IntervalType.NO_DATA,
                    IntervalType.POSITIVE,
                    IntervalType.NOT_APPLICABLE,
                ]
            ):
                df[f"p_i_{group_name}"] = elementwise_and(
                    df[f"p_{group_name}"], df[f"i_{group_name}"]
                )

        df["p"] = reduce(
            elementwise_or,
            [
                df[c]
                for c in df.columns
                if c.startswith("p_") and not c.startswith("p_i_")
            ],
        )
        df["i"] = reduce(
            elementwise_or, [df[c] for c in df.columns if c.startswith("i_")]
        )
        df["p_i"] = reduce(
            elementwise_or, [df[c] for c in df.columns if c.startswith("p_i_")]
        )

        assert len(df_base) == len(df)

        # &-combine all criteria with the base criterion to make sure that each criterion is only valid when the base
        # criterion is valid
        df = pd.merge(df_base, df, on=["person_id", "date"], how="left", validate="1:1")
        df = df.set_index(["person_id", "date"])
        df = df.apply(
            lambda x: elementwise_mask(x, df["BASE"], fill_value=IntervalType.NEGATIVE)
        )
        df = df.drop(columns="BASE")

        return df.reset_index()

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

        types = (
            df[["concept", "type"]]
            .drop_duplicates()
            .set_index("concept")["type"]
            .to_dict()
        )

        type_missing_data = {k: MISSING_DATA_TYPE[v] for k, v in types.items()}

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
        df_pivot[df_pivot.columns[2:]] = df_pivot[df_pivot.columns[2:]].map(
            lambda x: x > 0
        )

        # Create an auxiliary DataFrame with all combinations of person_id and dates between observation_start_date
        # and observation_end_date
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

        # Fill missing values with the missing data type
        for column in merged_df.columns[2:]:
            idx = merged_df[column]
            merged_df[column] = merged_df[column].astype(object)
            merged_df.loc[~idx, column] = type_missing_data[column]
            merged_df.loc[idx, column] = IntervalType.POSITIVE

        return merged_df

    @staticmethod
    def recommendation_test_runner(
        recommendation_url: str,
        observation_window: TimeRange,
        criteria_extended: pd.DataFrame,
    ) -> None:
        from execution_engine.clients import omopdb
        from execution_engine.execution_engine import ExecutionEngine

        e = ExecutionEngine(verbose=False)

        print(recommendation_url)
        recommendation = e.load_recommendation(recommendation_url, force_reload=False)

        e.execute(
            recommendation,
            start_datetime=observation_window.start,
            end_datetime=observation_window.end,
            use_multiprocessing=False,
        )

        def get_query(t, category):
            return (
                select(
                    t.c.run_id,
                    t.c.person_id,
                    PopulationInterventionPair.pi_pair_name,
                    t.c.cohort_category,
                    t.c.valid_date,
                )
                .outerjoin(PopulationInterventionPair)
                .where(t.c.criterion_id.is_(None))
                .where(t.c.cohort_category.in_(category))
            )

        def process_result(df_result):
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
                if row["pi_pair_name"] is None
                else f"{row['name']}_{row['pi_pair_name']}",
                axis=1,
            )

            df_result = df_result.rename(columns={"valid_date": "date"})

            df_result = df_result.pivot_table(
                columns="name",
                index=["person_id", "date"],
                values="run_id",
                aggfunc=len,
                fill_value=0,
            ).astype(bool)

            return df_result

        # P is fulfilled if they are fulfilled on any time of the day
        df_result_p_i = omopdb.query(
            get_query(
                partial_day_coverage,
                category=[CohortCategory.BASE, CohortCategory.POPULATION],
            )
        )

        # P_I is fulfilled only if it is fulfilled on the full day
        df_result_pi = omopdb.query(
            get_query(
                full_day_coverage,
                category=[
                    CohortCategory.INTERVENTION,
                    CohortCategory.POPULATION_INTERVENTION,
                ],
            )
        )

        df_result = pd.concat([df_result_p_i, df_result_pi])
        df_result = process_result(df_result)

        result_expected = RecommendationCriteriaCombination(
            name="expected", df=criteria_extended
        )
        result_db = result_expected.derive_database_result(df=df_result)

        assert result_db == result_expected
