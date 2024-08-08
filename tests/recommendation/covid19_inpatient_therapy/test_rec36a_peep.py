import copy

import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.interval import IntervalType
from tests._testdata.generator.data_generator import (
    BaseDataGenerator,
    MeasurementGenerator,
)
from tests._testdata.generator.generators import (
    COVID19,
    PEEP_5,
    PEEP_8,
    PEEP_10,
    PEEP_14,
    PEEP_18,
    FiO2_30,
    FiO2_40,
    FiO2_50,
    FiO2_60,
    FiO2_70,
    FiO2_80,
    FiO2_90,
    FiO2_100,
    Ventilated,
)
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2


@pytest.mark.recommendation
class TestRecommendation36aPeepV2(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-peep"
    )
    recommendation_package_version = "latest"
    recommendation_expression = {
        "PEEP_Intervention_Plan_FiO2_0.3": {
            "population": COVID19() & Ventilated() & FiO2_30(),
            # need to specify all higher values too because otherwise the expected values are incorrectly inferred
            "intervention": PEEP_5() | PEEP_8() | PEEP_10() | PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_30() & PEEP_5(),
        },
        "PEEP_Intervention_Plan_FiO2_0.4": {
            "population": COVID19() & Ventilated() & FiO2_40(),
            "intervention": PEEP_5() | PEEP_8() | PEEP_10() | PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_40() & PEEP_5(),
        },
        "PEEP_Intervention_Plan_FiO2_0.5": {
            "population": COVID19() & Ventilated() & FiO2_50(),
            "intervention": PEEP_8() | PEEP_10() | PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_50() & PEEP_8(),
        },
        "PEEP_Intervention_Plan_FiO2_0.6": {
            "population": COVID19() & Ventilated() & FiO2_60(),
            "intervention": PEEP_10() | PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_60() & PEEP_10(),
        },
        "PEEP_Intervention_Plan_FiO2_0.7": {
            "population": COVID19() & Ventilated() & FiO2_70(),
            "intervention": PEEP_10() | PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_70() & PEEP_10(),
        },
        "PEEP_Intervention_Plan_FiO2_0.8": {
            "population": COVID19() & Ventilated() & FiO2_80(),
            "intervention": PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_80() & PEEP_14(),
        },
        "PEEP_Intervention_Plan_FiO2_0.9": {
            "population": COVID19() & Ventilated() & FiO2_90(),
            "intervention": PEEP_14() | PEEP_18(),
            "population_intervention": COVID19() & Ventilated() & FiO2_90() & PEEP_14(),
        },
        "PEEP_Intervention_Plan_FiO2_1.0": {
            # need to specify all higher values too because otherwise the expected values are incorrectly inferred
            "population": COVID19() & Ventilated() & FiO2_100(),
            "intervention": PEEP_18(),
            "population_intervention": COVID19()
            & Ventilated()
            & FiO2_100()
            & PEEP_18(),
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All PEEP criteria (each optional)
        COVID19() | Ventilated() | FiO2_30() | PEEP_5(),
        COVID19() | Ventilated() | FiO2_30() | PEEP_8(),
        COVID19() | Ventilated() | FiO2_30() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_40() | PEEP_5() | PEEP_8(),
        COVID19() | Ventilated() | FiO2_50() | PEEP_8() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_60() | PEEP_10() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_70() | PEEP_10() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_80() | PEEP_14() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_90() | PEEP_14() | PEEP_18(),
        COVID19() | Ventilated() | FiO2_100() | PEEP_18(),
        (COVID19() | Ventilated()) & FiO2_30()
        # & FiO2_40()
        & (PEEP_5() | PEEP_8() | PEEP_18()),
        # (COVID19() | Ventilated()) & FiO2_50() & FiO2_60() & (PEEP_8() | PEEP_10()),
        # (COVID19() | Ventilated()) & FiO2_70() & FiO2_80() & (PEEP_10() | PEEP_14()),
        # (COVID19() | Ventilated()) & FiO2_90() & FiO2_100() & (PEEP_14() | PEEP_18()),
        (COVID19() | Ventilated())
        # & FiO2_30()
        # & FiO2_40()
        # & FiO2_50()
        & FiO2_60() & (PEEP_5() | PEEP_8() | PEEP_10() | PEEP_18()),
    ]

    """
    The following combinations (all pairwise different PEEP) are invalid, because when more than one PEEP is present,
    the higher PEEP criterion will be NEGATIVE during the time of the lower PEEP value, leading to an overall
    NEGATIVE result for the (higher) PEEP criterion, which is unexpected (by the test logic).
    """
    invalid_combinations = (
        (PEEP_18() & PEEP_14())
        | (PEEP_18() & PEEP_10())
        | (PEEP_18() & PEEP_8())
        | (PEEP_18() & PEEP_5())
        | (PEEP_14() & PEEP_10())
        | (PEEP_14() & PEEP_8())
        | (PEEP_14() & PEEP_5())
        | (PEEP_10() & PEEP_8())
        | (PEEP_10() & PEEP_5())
        | (PEEP_8() & PEEP_5())
    )

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        # NO_DATA FiO2 cols need to be set to "NEGATIVE", if any other FiO2 col is POSITIVE

        prefixes = ["FiO2_", "PEEP_"]
        for prefix in prefixes:
            cols = [c for c in df.columns if c.startswith(prefix)]
            idx_any_positive = (df[cols] == IntervalType.POSITIVE).any(axis=1)

            for c in cols:
                idx_no_data = df[c] == IntervalType.POSITIVE
                idx_no_data = idx_no_data.groupby("person_id").shift(
                    1, fill_value=False
                )  # shift by one to keep the first positive entry
                df.loc[idx_any_positive & idx_no_data, c] = IntervalType.NEGATIVE

        return df

    def _insert_criteria_hook(self, generator: BaseDataGenerator, data: list):
        if isinstance(generator, MeasurementGenerator) and (
            generator.name.startswith("FiO2_") or generator.name.startswith("PEEP_")
        ):
            # add another entry to make sure that FiO2 measurements are not valid longer than one hour,
            # as otherwise the determination of the expected data is very complex
            entry = sorted(data, key=lambda x: x.measurement_datetime)[-1]

            new_entry = copy.deepcopy(entry)
            if generator.name.startswith("FiO2_"):
                new_entry.measurement_datetime = (
                    entry.measurement_datetime + pd.Timedelta(minutes=59)
                )
            elif generator.name.startswith("PEEP_"):
                # set to the next day at midnight
                new_entry.measurement_datetime = new_entry.measurement_datetime.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + pd.Timedelta(days=1)

            new_entry.value_as_number = -99999

            data.append(new_entry)

    def test_recommendation_36a_peep(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
