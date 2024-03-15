import itertools
from functools import reduce

import numpy as np
import pandas as pd
import pytest

from execution_engine.omop.db.omop.tables import Person
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import TimeRange
from tests._testdata import concepts
from tests._testdata.generator.composite import AndGenerator, OrGenerator
from tests._testdata.generator.data_generator import BaseDataGenerator
from tests.functions import create_visit
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.utils.dataframe_operations import (
    combine_dataframe_via_logical_expression,
    elementwise_and,
    elementwise_or,
)


def generate_combinations(generator):
    gens = []

    if isinstance(generator, BaseDataGenerator):
        return [{generator: True}]

    elif isinstance(generator, AndGenerator) or isinstance(generator, OrGenerator):
        for gen in generator.generators:
            if isinstance(gen, BaseDataGenerator):
                if isinstance(generator, AndGenerator):
                    gens.append([{gen: True}])
                else:  # OrGenerator logic simplified
                    gens.append([{gen: True}, {gen: False}])
            elif isinstance(gen, (AndGenerator, OrGenerator)):
                sub_combinations = generate_combinations(gen)
                gens.append(sub_combinations)
            else:
                raise NotImplementedError(f"Unsupported generator type: {type(gen)}")

        # Calculate the cartesian product of combinations
        all_combinations = []
        for combination in itertools.product(*gens):
            combined_dict = {}
            for dict_ in combination:
                # print(combined_dict, dict_)
                combined_dict.update(dict_)
            all_combinations.append(combined_dict)
        return all_combinations


class TestRecommendationBaseV2(TestRecommendationBase):
    def insert_criteria_into_database(
        self, db_session, combinations: list[dict[BaseDataGenerator, bool]]
    ):
        """
        Inserts criteria entries from a DataFrame into the database, associating them with a newly created Person and
        Visit objects.
        """

        for person_id, combination in enumerate(combinations):
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
                visit_start_datetime=self.visit_datetime.start,
                visit_end_datetime=self.visit_datetime.end,
                visit_concept_id=concepts.INPATIENT_VISIT,
            )
            db_session.add_all([p, vo])

            for generator, valid in combination.items():
                db_session.add_all(generator.generate_data(vo, valid))

                # todo: add hooks
                # self._insert_criteria_hook(person_entries, entry, row)

            db_session.commit()

    def generate_criterion_entries(
        self, combinations: list[dict[BaseDataGenerator, bool]]
    ) -> pd.DataFrame:
        dfs = []

        for person_id, combination in enumerate(combinations):
            vo = create_visit(
                person_id=person_id,
                visit_start_datetime=self.visit_datetime.start,
                visit_end_datetime=self.visit_datetime.end,
                visit_concept_id=concepts.INPATIENT_VISIT,
            )

            for generator, valid in combination.items():
                data = generator.to_dict(vo=vo, valid=valid)
                df = pd.DataFrame(data).assign(person_id=person_id, name=str(generator))
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        return df

    @staticmethod
    def expand_dataframe_to_daily_observations(
        df: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """ """
        df = df.copy()

        # Set end_datetime equal to start_datetime if it's NaT
        df["end_datetime"] = pd.to_datetime(
            df["end_datetime"].fillna(df["start_datetime"]), utc=True
        )
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)

        # Vectorized expansion of DataFrame
        df["key"] = 1  # Create a key for merging
        date_range = pd.date_range(
            observation_window.start.date(), observation_window.end.date(), freq="D"
        )
        dates_df = pd.DataFrame({"date": date_range, "key": 1})
        df_expanded = df.merge(dates_df, on="key").drop("key", axis=1)
        df_expanded = df_expanded[
            df_expanded["date"].between(
                df_expanded["start_datetime"].dt.date,
                df_expanded["end_datetime"].dt.date,
            )
        ]

        types_missing_data = df_expanded.set_index("name")[
            "missing_data_type"
        ].to_dict()

        df_expanded.drop(
            ["start_datetime", "end_datetime", "missing_data_type"],
            axis=1,
            inplace=True,
        )

        # Pivot operation (remains the same if already efficient)
        df_pivot = df_expanded.pivot_table(
            index=["person_id", "date"],
            columns=["name"],
            aggfunc=len,
            fill_value=0,
        )

        # Reset index to make 'person_id' and 'date' regular columns
        df_pivot.reset_index(inplace=True)

        df_pivot.columns.name = None

        # Efficiently map and fill missing values
        df_pivot.iloc[:, 2:] = df_pivot.iloc[:, 2:].gt(0).astype(int)

        # Efficient merge with an auxiliary DataFrame
        aux_df = pd.DataFrame(
            itertools.product(df["person_id"].unique(), date_range),
            columns=["person_id", "date"],
        )
        df_pivot = df_pivot.sort_index(
            axis=1
        )  # Sort columns to avoid performance warning
        merged_df = pd.merge(aux_df, df_pivot, on=["person_id", "date"], how="left")
        merged_df.set_index(["person_id", "date"], inplace=True)

        # merged_df.columns = pd.MultiIndex.from_tuples(merged_df.columns)
        output_df = merged_df.copy()

        # Apply types_missing_data
        for column in merged_df.columns:
            idx_positive = merged_df[column].astype(bool) & merged_df[column].notnull()
            output_df[column] = types_missing_data[column]
            output_df.loc[idx_positive, column] = IntervalType.POSITIVE

        assert len(output_df.columns) == len(merged_df.columns), "Column count changed"

        return output_df

    def assemble_daily_recommendation_evaluation(
        self,
        df_entries: pd.DataFrame,
    ) -> pd.DataFrame:
        """ """
        idx_static = df_entries["static"]
        df_entries.loc[idx_static, "start_datetime"] = self.observation_window.start
        df_entries.loc[idx_static, "end_datetime"] = self.observation_window.end

        df = self.expand_dataframe_to_daily_observations(
            df_entries[
                [
                    "person_id",
                    "name",
                    "start_datetime",
                    "end_datetime",
                    "missing_data_type",
                ]
            ],
            observation_window=self.observation_window,
        )

        # the base criterion is the visit, all other criteria are AND-combined with the base criterion
        df_base = self.expand_dataframe_to_daily_observations(
            pd.DataFrame(
                {
                    "person_id": df_entries["person_id"].unique(),
                    "start_datetime": self.visit_datetime.start,
                    "end_datetime": self.visit_datetime.end,
                    "name": "BASE",
                    "missing_data_type": IntervalType.NEGATIVE,
                }
            ),
            self.observation_window,
        )

        # todo : add missing data ?
        # todo: modify criteria hook
        # df = self._modify_criteria_hook(df)

        for group_name, group in self.recommendation_expression.items():
            df[(f"p_{group_name}", "")] = combine_dataframe_via_logical_expression(
                df, group["population"]
            )
            df[(f"i_{group_name}", "")] = combine_dataframe_via_logical_expression(
                df, group["intervention"]
            )

            # expressions like "Eq(a+b+c, 1)" (at least one criterion) yield boolean columns and must
            # be converted to IntervalType
            if df[(f"p_{group_name}", "")].dtype == bool:
                df[(f"p_{group_name}", "")] = df[(f"p_{group_name}", "")].map(
                    {False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE}
                )

            if df[(f"i_{group_name}", "")].dtype == bool:
                df[(f"i_{group_name}", "")] = df[(f"i_{group_name}", "")].map(
                    {False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE}
                )

            if "population_intervention" in group:
                df[
                    (f"p_i_{group_name}", "")
                ] = combine_dataframe_via_logical_expression(
                    df, group["population_intervention"]
                )
            else:
                df[(f"p_i_{group_name}", "")] = elementwise_and(
                    df[(f"p_{group_name}", "")], df[(f"i_{group_name}", "")]
                )

        df[("p", "")] = reduce(
            elementwise_or,
            [
                df[c]
                for c in df.columns
                if c[0].startswith("p_") and not c[0].startswith("p_i_")
            ],
        )

        df[("i", "")] = reduce(
            elementwise_or, [df[c] for c in df.columns if c[0].startswith("i_")]
        )

        df[("p_i", "")] = reduce(
            elementwise_or, [df[c] for c in df.columns if c[0].startswith("p_i_")]
        )

        assert len(df_base) == len(df)

        # &-combine all criteria with the base criterion to make sure that each criterion is only valid when the base
        # criterion is valid
        df = pd.merge(df_base, df, on=["person_id", "date"], how="left", validate="1:1")

        mask = df[("BASE", "")].astype(bool)
        fill_value = np.repeat(np.array(IntervalType.NEGATIVE, dtype=object), len(df))
        df = df.apply(lambda x: np.where(mask, x, fill_value))

        df = df.drop(columns=("BASE", ""))

        return df.reset_index()

    @pytest.fixture(scope="function", autouse=True)
    def setup_testdata(self, db_session, run_slow_tests):
        combinations = [
            item for c in self.combinations for item in generate_combinations(c)
        ]
        # df_combinations is dataframe (binary) of all combinations that are to be performed (just by name)
        #   rows = persons, columns = criteria

        # df_combinations = pd.DataFrame(combinations).reset_index().rename(columns={"index": "person_id"})

        self.insert_criteria_into_database(db_session, combinations)

        df_criterion_entries = self.generate_criterion_entries(combinations)

        df_expected = self.assemble_daily_recommendation_evaluation(
            df_criterion_entries
        )

        yield df_expected
