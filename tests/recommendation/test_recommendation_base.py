import datetime
import itertools
from abc import ABC
from functools import reduce

import numpy as np
import pandas as pd
import pendulum
import pytest
from sqlalchemy import select

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
from tests.recommendation.utils.dataframe_operations import (
    combine_dataframe_via_logical_expression,
    elementwise_and,
    elementwise_or,
)
from tests.recommendation.utils.expression_parsing import (
    CRITERION_PATTERN,
    criteria_combination_str_to_df,
)
from tests.recommendation.utils.result_comparator import ResultComparator

MISSING_DATA_TYPE = {
    "condition": IntervalType.NEGATIVE,
    "observation": IntervalType.NO_DATA,
    "drug": IntervalType.NEGATIVE,
    "visit": IntervalType.NEGATIVE,
    "measurement": IntervalType.NO_DATA,
    "procedure": IntervalType.NEGATIVE,
}


@pytest.mark.recommendation
class TestRecommendationBase(ABC):
    recommendation_base_url = (
        "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
    )
    """
    The base URL of the canonical addresses (URLs) of the FHIR recommendation instances.

    Example:
        "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
    """

    visit_datetime = TimeRange(
        start="2023-03-01 07:00:00+01:00",
        end="2023-03-31 22:00:00+01:00",
        name="visit",
    )
    """
    An instance of TimeRange that specifies the start and end datetimes of a patient's visit in the context of this
    test.

    Example:
        TimeRange(
            start="2023-03-01 07:00:00+01:00",
            end="2023-03-31 22:00:00+01:00",
            name="visit",
        )
    """

    observation_window = TimeRange(
        start=visit_datetime.start - datetime.timedelta(days=3),
        end=visit_datetime.end + datetime.timedelta(days=3),
        name="observation",
    )
    """
    An instance of TimeRange that defines the time windows that is used when evaluating guideline adherence. For
    each day in the observation window, the guideline adherence is evaluated based on the data available on that day.

    Example:
        TimeRange(
            start=visit_datetime.start - datetime.timedelta(days=3),
            end=visit_datetime.end + datetime.timedelta(days=3),
            name="observation",
        )
    """

    recommendation_url = None
    """
    The URL or identifier of the Recommendation FHIR resource of the specific recommendation being considered.
    It is appended to the `recommendation_base_url` to form the full URL of the recommendation.

    Example:
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    """

    recommendation_package_version = None
    """
    The version of the recommendation FHIR package being used.Required to allow different versions of the
    recommendation package to be tested.

    Example:
        "v1.4.0-snapshot"
    """

    recommendation_expression = None
    """
    The symbolic expression of the recommendation, specifying the population and intervention criteria.

    Each key in the dictionary is a population-intervention pair identifier, and its value is another dictionary
    specifying the population criteria (key="population") and the intervention criteria (key="intervention").

    The 'population' key defines the eligibility criteria for the recommendation, using logical
    expressions to include or exclude certain conditions.

    The 'intervention' key defines the specific actions or treatments recommended, also using
    logical expressions to specify combinations or exclusions of treatments.

    Criteria naming
    ---------------
    * Individual criteria are represented as uppercase strings, which must be present as a CategoryDefinition with the
      same name in tests/_testdata/parameters.py.
    * A comparator (>, <, =) can be appended to the criterion name to modify the numerical value that is inserted
      into the database. Examples:
        * "DALTEPARIN=": The value of DALTEPARIN is inserted as is into the database.
        * "DALTEPARIN>": The value of DALTEPARIN is incremented by 1 before insertion into the database.
        * "DALTEPARIN<": The value of DALTEPARIN is decremented by 1 before insertion into the database.

    Example:
        {
            "AntithromboticProphylaxisWithLWMH": {
                "population": "COVID19 & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
                "intervention": "..."
            },
            ...
        }
    """

    combinations: list[str | None] = None
    """
    Specifies the criteria combinations for recommendation evaluation.

    `combinations`: Can take two types of values:
      1. `None`: Indicates that all possible combinations of criteria listed in the 'recommendation_expression'
                 variable are to be used. Note: This approach is exponential in the number of combinations.
      2. `List[str]`: A list where each string defines a specific set of criteria combinations to be included or
                      excluded in the evaluation. These strings must be formatted according to the
                      'criteria_combination_str_to_df' function, detailing criteria with their names and comparators.
                      Criteria can be marked as always present (no prefix), always absent ('!'), or optional ('?').

    String Format Details:
    - Criteria Representation: Each criterion within a string is represented by its
      name followed by a comparator ('<', '>', '=').
        - Prefix '!' indicates the criterion must always be absent in the combination.
        - Prefix '?' marks the criterion as optional.
        - Criteria without a prefix are considered always present.
    - Multiple Criteria: Criteria are separated by spaces.
    - The criterion names and comparators must match those in the `recommendation_expression` variable.

    Example:
        combinations = ["A>= !B<= ?C=", "!D> E< ?F="]
        # This specifies two combinations to be evaluated:
        # 1. 'A' must be present and at least as great as, 'B' must be absent, 'C' is optional.
        # 2. 'D' must be absent and greater than, 'E' must be present and less than, 'F' is optional.
    """

    invalid_combinations = ""
    """
    A logical expression representing combinations of criteria that are considered invalid or
    contradictory within the context of the recommendation. This string is used to filter out
    or prevent the application of the recommendation in scenarios where these invalid combinations
    are present.

    Example:
        "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"
    """

    def generate_criteria_combinations(
        self, criteria: set[str], run_slow_tests: bool
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of criteria combinations, filtering out invalid combinations and optionally limiting the
        output for quicker tests.

        This function creates a DataFrame that lists all possible combinations of provided criteria as binary factors.
        It then filters out any combinations deemed invalid according to a predefined logic or expression.
        If the function is instructed to run in a 'fast' mode (by setting `run_slow_tests` to False), the resulting
        DataFrame is further limited to a subset of combinations to speed up processing,
        particularly useful in testing environments where full combinatorial exploration may be time-consuming.

        Parameters:
            criteria (set[str]): A set of criteria names that will be used to generate binary combinations. Each
                                 criterion represents a column in the resulting DataFrame, with rows indicating the
                                 presence (True) or absence (False) of each criterion in a given combination.
            run_slow_tests (bool): A flag indicating whether to run slow tests. If False, the output DataFrame is
                                   limited to a representative subset (the first 100 and last 100 rows) of all possible
                                   combinations to speed up processing.

        Returns:
            pd.DataFrame: A DataFrame containing all valid combinations of the input criteria. Each row represents a
                          unique combination, with True/False values indicating the presence/absence of each criterion.
        """
        df = generate_binary_combinations_dataframe(criteria)

        # Remove invalid combinations
        if self.invalid_combinations:
            idx_invalid = combine_dataframe_via_logical_expression(
                df, self.invalid_combinations
            )
            df = df[~idx_invalid].copy()

        if not run_slow_tests:
            n = 50
            top, mid, bottom = df.head(n), df.iloc[n:-n], df.tail(n)

            df = pd.concat(
                (
                    top,
                    mid.sample(n=min(n * 2, len(mid)), replace=False, random_state=42),
                    bottom,
                )
            ).drop_duplicates()

        return df

    def generate_criterion_entries_from_criteria(
        self,
        df_criteria_combinations: pd.DataFrame,
    ):
        """
        Generates a DataFrame of criterion entries based on specified criteria combinations.

        This function iterates through a DataFrame of criteria combinations, evaluating each combination
        against a set of predefined criteria to generate detailed entries for each person. These entries
        include information such as the type of criterion, concept names, concept IDs, and relevant
        date-time information indicating when each criterion is applicable.

        Parameters:
            df_criteria_combinations (pd.DataFrame): A DataFrame where each row represents a combination
                of criteria for a person. Columns correspond to specific criteria, and rows are indexed
                by person IDs. The DataFrame should include boolean flags indicating the presence or
                absence of each criterion for each person.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an entry for a specific criterion applied
                to a person. The columns include 'person_id', 'type', 'concept', 'concept_id', 'static',
                'start_datetime', 'end_datetime', among other criterion-specific fields. The DataFrame
                aggregates these entries across all persons and criteria, taking into account any specified
                datetime offsets and handling specific criterion types with tailored logic.
        """
        entries = []

        def row_any(row, criterion_name):
            if criterion_name in row:
                return row[criterion_name].any()
            return False

        for person_id, row in df_criteria_combinations.iterrows():
            for criterion_name, comparator in self.get_criteria_name_and_comparator():
                criterion: parameter.CriterionDefinition = getattr(
                    parameter, criterion_name
                )

                if not row[(criterion_name, comparator)]:
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
                    "concept": criterion_name,
                    "comparator": comparator,
                    "concept_id": criterion.concept_id,
                    "static": criterion.static,
                }

                if criterion.type == "condition":
                    entry["start_datetime"] = self.visit_datetime.start
                    entry["end_datetime"] = self.visit_datetime.end
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
                    entry["route_concept_id"] = criterion.route_concept_id

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

            if row_any(row, "NADROPARIN_HIGH_WEIGHT") or row_any(
                row, "NADROPARIN_LOW_WEIGHT"
            ):
                entry_weight = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "WEIGHT",
                    "comparator": "=",
                    "concept_id": concepts.BODY_WEIGHT,
                    "start_datetime": datetime.datetime.combine(
                        self.visit_datetime.start.date(), datetime.time()
                    )
                    + datetime.timedelta(days=1),
                    "value": 71 if row_any(row, "NADROPARIN_HIGH_WEIGHT") else 69,
                    "unit_concept_id": concepts.UNIT_KG,
                    "static": True,
                }
                entries.append(entry_weight)
            elif row_any(row, "HEPARIN") or row_any(row, "ARGATROBAN"):
                entry_appt = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "APTT",
                    "comparator": ">",
                    "concept_id": concepts.LAB_APTT,
                    "start_datetime": entry["start_datetime"]
                    + datetime.timedelta(days=1),
                    "value": 51,
                    "unit_concept_id": concepts.UNIT_SECOND,
                    "static": False,
                }
                entries.append(entry_appt)
            elif row_any(row, "TIDAL_VOLUME"):
                # need to add height to calculate ideal body weight and then tidal volume per kg
                entry_weight = {
                    "person_id": person_id,
                    "type": "measurement",
                    "concept": "HEIGHT",
                    "comparator": "=",
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

    def insert_criteria_into_database(self, db_session, df_entries: pd.DataFrame):
        """
        Inserts criteria entries from a DataFrame into the database, associating them with a newly created Person and
        Visit objects.

        This function iterates over a DataFrame containing entries for various criteria (e.g., conditions, observations)
        and inserts them into a database. Each entry is associated with a Person and a Visit object, which are created
        based on the information in the DataFrame and some default values.

        Parameters:
            db_session: The database session used to insert the entries.
            df_entries (pd.DataFrame): A DataFrame containing the criteria entries. Each row represents an entry,
                with columns for 'person_id', 'type', 'concept_id', etc.
        """

        for person_id, g in df_entries.groupby("person_id"):
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
                        route_concept_id=row["route_concept_id"]
                        if not pd.isna(row["route_concept_id"])
                        else None,
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

                self._insert_criteria_hook(person_entries, entry, row)

                person_entries.append(entry)

            db_session.add_all(person_entries)
            db_session.commit()

    def _insert_criteria_hook(self, person_entries, entry, row):
        """
        A hook method intended for subclass overriding, providing a way to customize the behavior of
        `insert_criteria_into_database`.

        This method is called for each criteria entry before it is added to the list of entries to be inserted into the
        database. Subclasses can override this method to implement custom logic, such as filtering certain entries or
        modifying them before insertion.

        Parameters:
            person_entries: The list of entries (including Person and Visit objects) to be inserted into the database
                for a specific person.
            entry: The current entry being processed, which can be a condition, observation, measurement, drug exposure,
                or procedure object.
            row (pd.Series): The row from the DataFrame corresponding to the current entry, containing all relevant data
                or creating the entry.
        """

    def get_criteria_name_and_comparator(self) -> list[tuple[str, str]]:
        """
        Extracts criteria names and their associated comparators from recommendation expressions.

        This function parses the 'population' and 'intervention' fields of each recommendation expression to extract
        criteria names and any associated comparators (>, =, <). The criteria names are expected to be in uppercase
        and can include underscores. If no comparator is explicitly associated with a criteria name, "=" is assumed as the default comparator.

        Returns:
            list[tuple[str, str]]: A list of tuples, where each tuple contains a criteria name and its comparator.
            If no comparator is associated with a criteria name, "=" is used as the default comparator.
        """
        criteria: list[tuple[str, str]] = sum(
            [
                [
                    (
                        name,
                        comparator if comparator else "=",
                    )  # Use "=" as default if comparator is empty
                    for name, comparator in CRITERION_PATTERN.findall(
                        plan["population"] + " " + plan["intervention"],
                    )
                ]
                for plan in self.recommendation_expression.values()
            ],
            [],
        )

        return sorted(list(set(criteria)))

    def unique_criteria_names(self) -> list[str]:
        """
        Generates a list of unique criteria names extracted from recommendation expressions.

        This function utilizes `get_criteria_name_and_comparator` to fetch all criteria names along with their
         omparators, then filters out the comparator information to provide a list of unique criteria names.

        Returns:
            list[str]: A list of unique criteria names extracted from the recommendation expressions. The order of
            names is preserved.
        """

        names = [c[0] for c in self.get_criteria_name_and_comparator()]
        return list(dict.fromkeys(names))  # order preserving

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A placeholder hook method for subclasses to modify the DataFrame containing criteria entries.

        This method provides a mechanism for subclasses to implement custom logic that modifies the DataFrame of
        criteria entries before further processing or analysis. By default, this method returns the DataFrame unchanged,
        serving as a pass-through. Subclasses can override this method to apply specific transformations, filtering, or
        enhancements to the criteria data.

        Parameters:
            df (pd.DataFrame): The DataFrame containing criteria entries to be potentially modified.

        Returns:
            pd.DataFrame: The potentially modified DataFrame.
        """
        return df

    def assemble_daily_recommendation_evaluation(
        self,
        df_entries: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assembles a daily evaluation DataFrame for recommendation criteria, incorporating additional logic for base
        criteria and group-specific population and intervention criteria.

        This function takes a DataFrame of criteria entries and expands it to a daily granularity, creating a row for
        each person and day within the observation window.

        The resulting DataFrame is a comprehensive view of recommendation criteria evaluations on a daily basis for
        each person.

        Parameters:
            df_entries (pd.DataFrame): The DataFrame containing entries for various criteria. Each entry includes
                the person ID, concept, start and end datetimes, and the type of criterion (e.g., condition).

        Returns:
            pd.DataFrame: A DataFrame where each row represents a person-day combination, with columns for each
                criterion evaluated. The DataFrame includes all population and intervention criteria combinations
                (based on the recommendation_expression) and overall evaluations like 'p', 'i', and 'p_i' which
                represent the combined evaluation of population, intervention, and their intersection, respectively.
        """
        idx_static = df_entries["static"]
        df_entries.loc[idx_static, "start_datetime"] = self.observation_window.start
        df_entries.loc[idx_static, "end_datetime"] = self.observation_window.end

        df = self.expand_dataframe_to_daily_observations(
            df_entries[
                [
                    "person_id",
                    "concept",
                    "comparator",
                    "start_datetime",
                    "end_datetime",
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
                    "concept": "BASE",
                    "comparator": "",
                    "type": "visit",
                }
            ),
            self.observation_window,
        )

        for criterion_name, comparator in self.get_criteria_name_and_comparator():
            col = criterion_name, comparator
            if col not in df.columns:
                criterion = getattr(parameter, criterion_name)
                if criterion.missing_data_type is not None:
                    df[col] = criterion.missing_data_type
                else:
                    df[col] = MISSING_DATA_TYPE[criterion.type]

        df = self._modify_criteria_hook(df)

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

    @staticmethod
    def expand_dataframe_to_daily_observations(
        df: pd.DataFrame, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Expands a DataFrame to detail daily observations for each person within a specified observation window.

        This function transforms an input DataFrame, which contains one row per person and one column per concept, into
        an expanded DataFrame where each row corresponds to a single day's observation of a person for the specified
        time range. The expansion is based on the `observation_window`, which defines the start and end dates for the
        observations.

        Parameters:
            df (pd.DataFrame): The input DataFrame with columns indicating different concepts and rows representing
                               individual observations per person.
            observation_window (TimeRange): An object with `start` and `end` attributes specifying the date range for
                                            which to expand the observations.

        Returns:
            pd.DataFrame: An expanded DataFrame where each row represents an observation for a person on a specific day
                          within the observation window. The DataFrame's columns correspond to the original concepts,
                          expanded to indicate the presence or absence of each concept per person per day.

        """
        df = df.copy()

        # Set end_datetime equal to start_datetime if it's NaT
        df["end_datetime"] = pd.to_datetime(
            df["end_datetime"].fillna(df["start_datetime"]), utc=True
        )
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)

        types = (
            df[["concept", "comparator", "type"]]
            .drop_duplicates()
            .set_index(["concept", "comparator"])["type"]
            .to_dict()
        )

        types_missing_data = {}
        for (parameter_name, comparator), parameter_type in types.items():
            if hasattr(parameter, parameter_name):
                criterion_def = getattr(parameter, parameter_name)
            else:
                criterion_def = None

            if (
                criterion_def is not None
                and criterion_def.missing_data_type is not None
            ):
                types_missing_data[parameter_name] = criterion_def.missing_data_type
            else:
                types_missing_data[parameter_name] = MISSING_DATA_TYPE[parameter_type]

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

        df_expanded.drop(
            ["start_datetime", "end_datetime", "type"], axis=1, inplace=True
        )

        # Pivot operation (remains the same if already efficient)
        df_pivot = df_expanded.pivot_table(
            index=["person_id", "date"],
            columns=["concept", "comparator"],
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
            columns=pd.MultiIndex.from_tuples([("person_id", ""), ("date", "")]),
        )
        df_pivot = df_pivot.sort_index(
            axis=1
        )  # Sort columns to avoid performance warning
        merged_df = pd.merge(aux_df, df_pivot, on=["person_id", "date"], how="left")
        merged_df.set_index(["person_id", "date"], inplace=True)

        merged_df.columns = pd.MultiIndex.from_tuples(merged_df.columns)
        output_df = merged_df.copy()

        # Apply types_missing_data
        for column in merged_df.columns:
            idx_positive = merged_df[column].astype(bool) & merged_df[column].notnull()
            output_df[column] = types_missing_data[column[0]]
            output_df.loc[idx_positive, column] = IntervalType.POSITIVE

        assert len(output_df.columns) == len(merged_df.columns), "Column count changed"

        return output_df

    @pytest.fixture(scope="function", autouse=True)
    def setup_testdata(self, db_session, run_slow_tests):
        criteria = self.get_criteria_name_and_comparator()

        if self.combinations is None:
            # df_combinations is dataframe (binary) of all combinations that are to be performed (just by name)
            #   rows = persons, columns = criteria
            df_combinations = self.generate_criteria_combinations(
                criteria, run_slow_tests=run_slow_tests
            )
        else:
            # df_combinations is dataframe (binary) of all combinations that are to be performed (just by name)
            #   rows = persons, columns = criteria
            df_combinations = []
            for combination_str in self.combinations:
                df_combination = criteria_combination_str_to_df(combination_str)
                # assert that each column name is in the criteria list
                for c in df_combination.columns:
                    if c not in criteria:
                        raise ValueError(
                            f"Criterion {''.join(c)} not in recommendation criteria"
                        )
                df_combinations.append(df_combination)
            df_combinations = (
                pd.concat(df_combinations).fillna(False).reset_index(drop=True)
            )

        assert set(df_combinations.columns) == set(
            criteria
        ), "All criteria must be present in test combinations"

        #   rows = single actual criteria (with date, value etc.)
        #   columns = person_id, type, concept, start_datetime, end_datetime, value
        df_criterion_entries = self.generate_criterion_entries_from_criteria(
            df_combinations
        )

        self.insert_criteria_into_database(db_session, df_criterion_entries)

        df_expected = self.assemble_daily_recommendation_evaluation(
            df_criterion_entries
        )

        yield df_expected

    def recommendation_test_runner(
        self,
        df_expected: pd.DataFrame,
    ) -> None:
        from execution_engine.clients import omopdb
        from execution_engine.execution_engine import ExecutionEngine

        e = ExecutionEngine(verbose=False)

        recommendation_url = self.recommendation_base_url + self.recommendation_url

        print(recommendation_url)
        recommendation = e.load_recommendation(
            recommendation_url,
            recommendation_package_version=self.recommendation_package_version,
            force_reload=False,
        )

        e.execute(
            recommendation,
            start_datetime=self.observation_window.start,
            end_datetime=self.observation_window.end,
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

        result_expected = ResultComparator(name="expected", df=df_expected)
        result_db = result_expected.derive_database_result(df=df_result)

        assert result_db == result_expected
