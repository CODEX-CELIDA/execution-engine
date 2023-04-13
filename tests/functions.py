import datetime
import itertools
import random
import re

import numpy as np
import pandas as pd

from execution_engine.omop.db.cdm import (
    ConditionOccurrence,
    DrugExposure,
    Measurement,
    Observation,
    Person,
    ProcedureOccurrence,
    VisitOccurrence,
)
from tests import concepts


def random_date(start_date: datetime.date, end_date: datetime.date) -> datetime.date:
    """Generate a random datetime between `start_date` and `end_date`"""
    return start_date + datetime.timedelta(
        days=random.randint(0, (end_date - start_date).days),
    )


def random_datetime(date: datetime.date, max_hours: int = 24) -> datetime.datetime:
    """Generate a random datetime between `date` and `date` + 1 day"""
    return datetime.datetime.combine(date, datetime.time()) + datetime.timedelta(
        seconds=random.randint(0, max_hours * 3600)
    )


def generate_dataframe(keys):
    # Create all possible combinations of binary factors
    options = [False, True]
    combinations = list(itertools.product(options, repeat=len(keys)))

    # Create a pandas DataFrame from the combinations
    df = pd.DataFrame(combinations, columns=keys).astype(bool)

    return df


def create_visit(p: Person, visit_start_date, visit_end_date, icu=True):
    return VisitOccurrence(
        person_id=p.person_id,
        visit_start_date=visit_start_date,
        visit_start_datetime=random_datetime(visit_start_date),
        visit_end_date=visit_end_date,
        visit_end_datetime=random_datetime(visit_end_date),
        visit_concept_id=concepts.INTENSIVE_CARE if icu else concepts.INPATIENT_VISIT,
        visit_type_concept_id=concepts.VISIT_TYPE_STILL_PATIENT,
    )


def create_condition(vo: VisitOccurrence, condition_concept_id):
    return ConditionOccurrence(
        vo.person_id,
        condition_concept_id=condition_concept_id,
        condition_start_date=vo.visit_start_date,
        condition_start_datetime=vo.visit_start_datetime,
        condition_end_date=vo.visit_end_date,
        condition_end_datetime=vo.visit_end_datetime,
        condition_type_concept_id=concepts.EHR,
    )


def create_drug_exposure(
    vo: VisitOccurrence, drug_concept_id, start_datetime, end_datetime, quantity
):
    assert start_datetime >= vo.visit_start_datetime
    assert end_datetime <= vo.visit_end_datetime
    assert start_datetime <= end_datetime

    return DrugExposure(
        person_id=vo.person_id,
        drug_concept_id=drug_concept_id,
        drug_exposure_start_datetime=start_datetime,
        drug_exposure_start_date=start_datetime.date(),
        drug_exposure_end_datetime=end_datetime,
        drug_exposure_end_date=end_datetime.date(),
        quantity=quantity,
        drug_type_concept_id=concepts.EHR,
    )


def create_measurement(
    vo: VisitOccurrence,
    measurement_concept_id,
    datetime,
    value_as_number,
    unit_concept_id,
):
    assert datetime >= vo.visit_start_datetime
    assert datetime <= vo.visit_end_datetime

    return Measurement(
        person_id=vo.person_id,
        measurement_concept_id=measurement_concept_id,
        measurement_date=datetime.date(),
        measurement_datetime=datetime,
        value_as_number=value_as_number,
        unit_concept_id=unit_concept_id,
        measurement_type_concept_id=concepts.EHR,
    )


def create_observation(vo: VisitOccurrence, observation_concept_id, datetime):
    assert datetime >= vo.visit_start_datetime
    assert datetime <= vo.visit_end_datetime

    return Observation(
        person_id=vo.person_id,
        observation_concept_id=observation_concept_id,
        observation_date=datetime.date(),
        observation_datetime=datetime,
        observation_type_concept_id=concepts.EHR,
    )


def create_procedure(
    vo: VisitOccurrence, procedure_concept_id, start_datetime, end_datetime
):
    assert start_datetime >= vo.visit_start_datetime
    assert end_datetime <= vo.visit_end_datetime
    assert start_datetime <= end_datetime

    return ProcedureOccurrence(
        person_id=vo.person_id,
        procedure_concept_id=procedure_concept_id,
        procedure_type_concept_id=concepts.EHR,
        procedure_date=start_datetime.date(),
        procedure_datetime=start_datetime,
        procedure_end_date=end_datetime.date(),
        procedure_end_datetime=end_datetime,
    )


def to_extended(df, observation_start_date, observation_end_date):
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
    date_range = pd.date_range(observation_start_date, observation_end_date, freq="D")
    aux_df = pd.DataFrame(
        {
            "person_id": np.repeat(unique_person_ids, len(date_range)),
            "date": np.tile(date_range, len(unique_person_ids)),
        }
    )

    # Merge the auxiliary DataFrame with the pivoted DataFrame
    merged_df = pd.merge(aux_df, df_pivot, on=["person_id", "date"], how="left")

    # Fill missing values with False
    merged_df[merged_df.columns[2:]] = merged_df[merged_df.columns[2:]].fillna(False)

    return merged_df


def to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()