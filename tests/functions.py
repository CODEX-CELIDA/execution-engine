import datetime
import itertools
import re

import pandas as pd

from execution_engine.omop.db.omop.tables import (
    ConditionOccurrence,
    DrugExposure,
    Measurement,
    Observation,
    ProcedureOccurrence,
    VisitDetail,
    VisitOccurrence,
)
from tests._testdata import concepts


def generate_binary_combinations_dataframe(keys: list[str]) -> pd.DataFrame:
    """
    Generate a pandas DataFrame with all possible combinations of binary factors
    """
    # Create all possible combinations of binary factors
    options = [False, True]
    combinations = list(itertools.product(options, repeat=len(keys)))

    # Create a pandas DataFrame from the combinations
    df = pd.DataFrame(combinations, columns=keys).astype(bool)

    return df


def create_visit(
    person_id: int,
    visit_start_datetime: datetime.datetime,
    visit_end_datetime: datetime.datetime,
    visit_concept_id: int,
) -> VisitOccurrence:
    """
    Create a visit for a person (one single encounter)
    """
    return VisitOccurrence(
        person_id=person_id,
        visit_start_date=visit_start_datetime.date(),
        visit_start_datetime=visit_start_datetime,
        visit_end_date=visit_end_datetime.date(),
        visit_end_datetime=visit_end_datetime,
        visit_concept_id=visit_concept_id,
        visit_type_concept_id=concepts.VISIT_TYPE_STILL_PATIENT,
    )


def create_visit_detail(
    vo: VisitOccurrence,
    visit_detail_start_datetime: datetime.datetime,
    visit_detail_end_datetime: datetime.datetime,
    visit_detail_concept_id: int,
) -> VisitDetail:
    """
    Create a visit detail for a person (e.g. transfer between units in the hospital)
    """
    return VisitDetail(
        person_id=vo.person_id,
        visit_detail_concept_id=visit_detail_concept_id,
        visit_detail_start_date=visit_detail_start_datetime.date(),
        visit_detail_start_datetime=visit_detail_start_datetime,
        visit_detail_end_date=visit_detail_end_datetime.date(),
        visit_detail_end_datetime=visit_detail_end_datetime,
        visit_detail_type_concept_id=concepts.EHR,
        visit_occurrence_id=vo.visit_occurrence_id,
    )


def create_condition(
    vo: VisitOccurrence,
    condition_concept_id: int,
    condition_start_datetime: datetime.datetime | None = None,
    condition_end_datetime: datetime.datetime | None = None,
) -> ConditionOccurrence:
    """
    Create a condition for a visit
    """

    start_datetime = (
        condition_start_datetime
        if condition_start_datetime is not None
        else vo.visit_start_datetime
    )
    end_datetime = (
        condition_end_datetime
        if condition_end_datetime is not None
        else vo.visit_end_datetime
    )

    return ConditionOccurrence(
        person_id=vo.person_id,
        condition_concept_id=condition_concept_id,
        condition_start_date=start_datetime.date(),
        condition_start_datetime=start_datetime,
        condition_end_date=end_datetime.date(),
        condition_end_datetime=end_datetime,
        condition_type_concept_id=concepts.EHR,
    )


def create_drug_exposure(
    vo: VisitOccurrence,
    drug_concept_id: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    quantity: float,
) -> DrugExposure:
    """
    Create a drug exposure for a visit
    """
    assert (
        start_datetime <= end_datetime
    ), "drug_exposure: start_datetime must be before end_datetime"

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
    measurement_concept_id: int,
    measurement_datetime: datetime.datetime,
    value_as_number: float | None = None,
    value_as_concept_id: int | None = None,
    unit_concept_id: int | None = None,
) -> Measurement:
    """
    Create a measurement for a visit
    """
    return Measurement(
        person_id=vo.person_id,
        measurement_concept_id=measurement_concept_id,
        measurement_date=measurement_datetime.date(),
        measurement_datetime=measurement_datetime,
        value_as_number=value_as_number,
        value_as_concept_id=value_as_concept_id,
        unit_concept_id=unit_concept_id,
        measurement_type_concept_id=concepts.EHR,
    )


def create_observation(
    vo: VisitOccurrence,
    observation_concept_id: int,
    observation_datetime: datetime.datetime,
    value_as_number: float | None = None,
    value_as_string: str | None = None,
    value_as_concept_id: int | None = None,
    unit_concept_id: int | None = None,
) -> Observation:
    """
    Create an observation for a visit
    """
    return Observation(
        person_id=vo.person_id,
        observation_concept_id=observation_concept_id,
        observation_date=observation_datetime.date(),
        observation_datetime=observation_datetime,
        observation_type_concept_id=concepts.EHR,
        value_as_number=value_as_number,
        value_as_string=value_as_string,
        value_as_concept_id=value_as_concept_id,
        unit_concept_id=unit_concept_id,
    )


def create_procedure(
    vo: VisitOccurrence,
    procedure_concept_id: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
) -> ProcedureOccurrence:
    """
    Create a procedure for a visit
    """
    assert (
        start_datetime <= end_datetime
    ), "procedure: start_datetime must be before end_datetime"

    return ProcedureOccurrence(
        person_id=vo.person_id,
        procedure_concept_id=procedure_concept_id,
        procedure_type_concept_id=concepts.EHR,
        procedure_date=start_datetime.date(),
        procedure_datetime=start_datetime,
        procedure_end_date=end_datetime.date(),
        procedure_end_datetime=end_datetime,
    )


def to_snake(s: str) -> str:
    """Converts a string from CamelCase to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def get_fraction_per_day(
    datetime_start: datetime.datetime, datetime_end: datetime.datetime
) -> dict[datetime.date, float]:
    """
    Get the fraction of the total time between `datetime_start` and `datetime_end` that falls on each day between
    `datetime_start` and `datetime_end`.
    """
    total_seconds = (datetime_end - datetime_start).total_seconds()
    current_datetime = datetime_start
    fractions = {}

    while current_datetime.date() <= datetime_end.date():
        next_datetime = (current_datetime + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0
        )
        seconds_this_day = (
            min(next_datetime, datetime_end) - current_datetime
        ).total_seconds()
        fractions[current_datetime.date()] = seconds_this_day / total_seconds
        current_datetime = next_datetime

    assert abs(sum(fractions.values()) - 1.0) < 0.00001, "Fractions do not sum up to 1"

    return fractions
