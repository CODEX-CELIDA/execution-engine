import itertools
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
import pytest

from execution_engine.omop.db.omop.tables import Person
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import TimeRange
from tests._testdata import concepts
from tests._testdata.generator.composite import (
    AndGenerator,
    AtLeastOneGenerator,
    CompositeDataGenerator,
    ExactlyOneGenerator,
    NotGenerator,
    OrGenerator,
)
from tests._testdata.generator.data_generator import BaseDataGenerator, VisitGenerator
from tests.functions import create_visit
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.utils.dataframe_operations import (
    elementwise_add,
    elementwise_and,
    elementwise_not,
    elementwise_or,
)


def contains_invalid_combinations(
    combination: dict[BaseDataGenerator, bool],
    invalid_combinations: OrGenerator | AndGenerator,
) -> bool:
    """
    Check if a combination of criteria contains any invalid combinations based on the provided invalid_combinations.
    """

    if not invalid_combinations:
        return False

    valid_criteria = [k for k, v in combination.items() if v]

    if isinstance(invalid_combinations, OrGenerator):
        for gen in invalid_combinations.generators:
            if contains_invalid_combinations(combination, gen):
                return True
    elif isinstance(invalid_combinations, AndGenerator):
        assert all(
            isinstance(gen, BaseDataGenerator)
            for gen in invalid_combinations.generators
        ), "Only BaseDataGenerator instances are supported in invalid_combinations AndGenerator"
        if set(invalid_combinations.generators) <= set(valid_criteria):
            return True
    elif isinstance(invalid_combinations, NotGenerator):
        # we just ignore the NOT operator
        return contains_invalid_combinations(
            combination, invalid_combinations.generator
        )
    else:
        raise NotImplementedError(
            f"Unsupported generator type: {type(invalid_combinations)}"
        )

    return False


def generate_combinations(generator, invalid_combinations, default_value: bool = True):
    gens = []

    if isinstance(generator, BaseDataGenerator):
        return [{generator: default_value}]

    elif isinstance(generator, (AndGenerator, OrGenerator)):
        for gen in generator.generators:
            if isinstance(gen, (BaseDataGenerator, NotGenerator)):
                if isinstance(gen, NotGenerator):
                    assert isinstance(
                        gen.generator, BaseDataGenerator
                    ), "Only BaseDataGenerator instances are supported in NotGenerator"
                    if isinstance(generator, AndGenerator):
                        gens.append([{gen.generator: not default_value}])
                    else:  # OrGenerator logic simplified
                        gens.append(
                            [
                                {gen.generator: default_value},
                                {gen.generator: not default_value},
                            ]
                        )
                else:
                    if isinstance(generator, AndGenerator):
                        gens.append([{gen: default_value}])
                    else:  # OrGenerator logic simplified
                        gens.append([{gen: default_value}, {gen: not default_value}])
            elif isinstance(gen, (AndGenerator, OrGenerator)):
                sub_combinations = generate_combinations(
                    gen, invalid_combinations, default_value
                )
                gens.append(sub_combinations)
            elif isinstance(gen, NotGenerator):
                # sub_combinations = generate_combinations(
                #     gen.generator, invalid_combinations, not default_value
                # )
                # gens.append(sub_combinations)
                raise NotImplementedError(
                    "NotGenerator inside AndGenerator/OrGenerator"
                )
            else:
                raise NotImplementedError(f"Unsupported generator type: {type(gen)}")

        # Calculate the cartesian product of combinations
        all_combinations = []
        for combination in itertools.product(*gens):
            combined_dict = {}
            for dict_ in combination:
                # print(combined_dict, dict_)
                combined_dict.update(dict_)
            if contains_invalid_combinations(combined_dict, invalid_combinations):
                continue
            all_combinations.append(combined_dict)
        return all_combinations


def evaluate_expression(
    expr: BaseDataGenerator | CompositeDataGenerator,
    df: pd.DataFrame,
    default_value: Any = IntervalType.NEGATIVE,
) -> pd.Series:
    """
    Evaluates a composite data generator expression on a pandas DataFrame and returns the result as a pandas Series.

    The function recursively traverses through the expression, which can be composed of BaseDataGenerator instances and
    CompositeDataGenerator instances (such as AndGenerator, OrGenerator, NotGenerator, etc.). It evaluates the expression
    by performing the corresponding logical operations on the DataFrame columns associated with each BaseDataGenerator.

    Parameters
    ----------
    expr : Union[BaseDataGenerator, CompositeDataGenerator]
     The composite data generator expression to evaluate. This can be a single BaseDataGenerator, representing a
     DataFrame column, or a CompositeDataGenerator, which combines multiple generators through logical operations.

    df : pd.DataFrame
     The pandas DataFrame on which the expression is to be evaluated. The DataFrame should contain columns corresponding
     to the names of the BaseDataGenerator instances within the expression. The column names are expected to match
     the `name` property of each BaseDataGenerator.

    Returns
    -------
    pd.Series
     A pandas Series representing the result of evaluating the expression on the DataFrame. The Series contains boolean
     values, where each value corresponds to a row in the DataFrame and indicates the result of the expression for that
     row.

    Raises
    ------
    NotImplementedError
     If the function encounters a CompositeDataGenerator subclass that hasn't been implemented in the evaluation logic.

    TypeError
     If the `expr` argument is not an instance of BaseDataGenerator or CompositeDataGenerator.

    Example
    -------
    >>> # Assuming Gen1, Gen2, Gen3, and Gen4 are defined BaseDataGenerator subclasses
    >>> expr = AndGenerator(Gen1(), Gen2(), OrGenerator(Gen3(), Gen4()))
    >>> df = pd.DataFrame({...})
    >>> result = evaluate_expression(expr, df)
    >>> print(result)

    Notes
    -----
    - The function supports logical AND (&), OR (|), and NOT (~) operations through AndGenerator, OrGenerator, and
    NotGenerator, respectively. Extend the function as necessary to include other types of composite generators.
    - The evaluation of ExactlyOneGenerator and AtLeastOneGenerator is based on counting the number of true values among
    the evaluated generators and applying the corresponding logic. Adjust these implementations as needed.
    """

    def eval_expr(x):
        return evaluate_expression(x, df)

    if isinstance(expr, BaseDataGenerator):
        # Base case: Return the corresponding DataFrame column
        key = str(expr)
        if key in df:
            return df[key]
        else:
            return pd.Series(index=df.index, name=key, data=default_value)
    elif isinstance(expr, CompositeDataGenerator):
        if isinstance(expr, AndGenerator):
            return reduce(elementwise_and, map(eval_expr, expr.generators))
        elif isinstance(expr, OrGenerator):
            return reduce(elementwise_or, map(eval_expr, expr.generators))
        elif isinstance(expr, NotGenerator):
            return elementwise_not(eval_expr(expr.generator))
        elif isinstance(expr, ExactlyOneGenerator):
            result = (
                reduce(
                    elementwise_add,
                    map(
                        lambda col: df[str(expr)].astype(bool).astype(int),
                        expr.generators,
                    ),
                )
                == 1
            ).map({False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE})

            return result
        elif isinstance(expr, AtLeastOneGenerator):
            result = (
                reduce(
                    elementwise_add,
                    map(
                        lambda col: df[str(expr)].astype(bool).astype(int),
                        expr.generators,
                    ),
                )
                >= 1
            ).map({False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE})

            return result

        else:
            raise NotImplementedError(f"Operation {type(expr)} is not implemented")
    else:
        raise TypeError(f"Unsupported type: {type(expr)}")


class TestRecommendationBaseV2(TestRecommendationBase):
    invalid_combinations = []

    recommendation_parser_version = 2

    def distinct_criteria(self) -> set[str]:
        criteria = set()

        for plan in self.recommendation_expression.values():
            for type_ in ["population", "intervention"]:
                if isinstance(plan[type_], CompositeDataGenerator):
                    criteria |= plan[type_].flatten()
                elif isinstance(plan[type_], BaseDataGenerator):
                    criteria.add(plan[type_])
                else:
                    raise ValueError(f"Invalid type {type(plan[type_])}")

        return criteria

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

            visit_gen_found = False
            for generator in combination:
                if isinstance(generator, VisitGenerator):
                    if visit_gen_found:
                        raise ValueError(
                            "Only one VisitGenerator is allowed per combination"
                        )
                    visit_gen_found = True
                    vo = generator.generate_data(
                        person_id, valid=combination[generator]
                    )
                    assert (
                        len(vo) == 1
                    ), "VisitGenerator must return exactly one VisitOccurrence"
                    vo = vo[0]
                    db_session.add(vo)

            # add a default visit if no visit generator was found
            # always add the default visit as well
            vo = create_visit(
                person_id=p.person_id,
                visit_start_datetime=self.visit_datetime.start,
                visit_end_datetime=self.visit_datetime.end,
                visit_concept_id=concepts.INPATIENT_VISIT,
            )
            db_session.add_all([p, vo])

            for generator, valid in combination.items():
                if isinstance(generator, VisitGenerator):
                    # skip the visit generator, as it has already been processed
                    continue

                data = generator.generate_data(vo, valid)

                # the generator may return no data to insert, e.g. in the FiO2 cases, where an invalid value of
                # one FiO2 interval (e.g. FiO2_30, from 30%-39.999% - where the invalid value would be 29% (=30-1))
                # would be a valid value for another interval(e.g. FiO2_20, from 20%-29.999%)
                if not data:
                    continue

                self._insert_criteria_hook(generator, data)

                db_session.add_all(data)

            db_session.commit()

    def _insert_criteria_hook(self, generator: BaseDataGenerator, data: list):
        """
        A hook method intended for subclass overriding, providing a way to customize the behavior of
        `insert_criteria_into_database`.

        This method is called for each criteria entry before it is added to the list of entries to be inserted into the
        database. Subclasses can override this method to implement custom logic, such as filtering certain entries or
        modifying them before insertion.
        """

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
                data = generator.to_dict(vo, valid=valid)
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

        types = df_expanded.set_index("name")["type"].to_dict()

        df_expanded.drop(
            ["start_datetime", "end_datetime", "missing_data_type", "type"],
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

            # Apply types
            if types[column] in ["measurement", "observation"]:
                # we need to forward fill the data
                s = output_df[column].copy()
                # Group by person_id and forward fill
                idx_no_data = s == IntervalType.NO_DATA
                s.loc[idx_no_data] = np.nan
                s = s.groupby(level="person_id").ffill()
                s.loc[s.isnull()] = IntervalType.NO_DATA
                # Replace np.nan back to 'NO_DATA' if needed
                output_df[column] = s

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

        df_entries = df_entries[df_entries["valid"]]

        df = self.expand_dataframe_to_daily_observations(
            df_entries[
                [
                    "person_id",
                    "name",
                    "start_datetime",
                    "end_datetime",
                    "missing_data_type",
                    "type",
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
                    "type": "visit",
                }
            ),
            self.observation_window,
        )

        criteria = self.distinct_criteria()

        missings = [c for c in criteria if str(c) not in df.columns]

        for gen in missings:
            df[str(gen)] = gen.missing_data_type

        df = self._modify_criteria_hook(df)

        for group_name, group in self.recommendation_expression.items():
            df[f"p_{group_name}"] = evaluate_expression(group["population"], df)
            df[f"i_{group_name}"] = evaluate_expression(group["intervention"], df)

            # filter intervention by population
            if df[f"i_{group_name}"].dtype == bool:
                df[f"i_{group_name}"] &= df[f"p_{group_name}"]
            else:
                df[f"i_{group_name}"] = (
                    df[f"p_{group_name}"] == IntervalType.POSITIVE
                ) & (df[f"i_{group_name}"] == IntervalType.POSITIVE)

            # expressions like "Eq(a+b+c, 1)" (at least one criterion) yield boolean columns and must
            # be converted to IntervalType
            if df[f"p_{group_name}"].dtype == bool:
                df[f"p_{group_name}"] = df[f"p_{group_name}"].map(
                    {False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE}
                )

            if df[f"i_{group_name}"].dtype == bool:
                df[f"i_{group_name}"] = df[f"i_{group_name}"].map(
                    {False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE}
                )

            if "population_intervention" in group:
                df[f"p_i_{group_name}"] = evaluate_expression(
                    group["population_intervention"], df
                )
            else:
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

        mask = df["BASE"].astype(bool)
        fill_value = np.repeat(np.array(IntervalType.NEGATIVE, dtype=object), len(df))
        df = df.apply(lambda x: np.where(mask, x, fill_value))

        df = df.drop(columns="BASE")

        return df.reset_index()

    @pytest.fixture(scope="function", autouse=True)
    def setup_testdata(self, db_session, run_slow_tests):
        combinations = [
            item
            for c in self.combinations
            for item in generate_combinations(c, self.invalid_combinations)
        ]

        # combinations = [combinations[0]]

        self.insert_criteria_into_database(db_session, combinations)

        df_criterion_entries = self.generate_criterion_entries(combinations)

        df_expected = self.assemble_daily_recommendation_evaluation(
            df_criterion_entries
        )

        yield df_expected
