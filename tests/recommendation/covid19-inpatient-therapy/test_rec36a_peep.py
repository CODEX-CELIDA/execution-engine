import copy

import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.interval import IntervalType
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation36aPeep(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-peep"
    )
    recommendation_package_version = "latest"
    recommendation_expression = {
        "PEEP_Intervention_Plan_FiO2_0.3": {
            "population": "COVID19 & VENTILATED & FiO2_30",
            # need to specify all higher values too because otherwise the expected values are incorrectly inferred
            "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_30 & PEEP_5>",
        },
        "PEEP_Intervention_Plan_FiO2_0.4": {
            "population": "COVID19 & VENTILATED & FiO2_40",
            "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_40 & PEEP_5>",
        },
        "PEEP_Intervention_Plan_FiO2_0.5": {
            "population": "COVID19 & VENTILATED & FiO2_50",
            "intervention": "PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_50 & PEEP_8>",
        },
        "PEEP_Intervention_Plan_FiO2_0.6": {
            "population": "COVID19 & VENTILATED & FiO2_60",
            "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_60 & PEEP_10>",
        },
        "PEEP_Intervention_Plan_FiO2_0.7": {
            "population": "COVID19 & VENTILATED & FiO2_70",
            "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_70 & PEEP_10>",
        },
        "PEEP_Intervention_Plan_FiO2_0.8": {
            "population": "COVID19 & VENTILATED & FiO2_80",
            "intervention": "PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_80 & PEEP_14>",
        },
        "PEEP_Intervention_Plan_FiO2_0.9": {
            "population": "COVID19 & VENTILATED & FiO2_90",
            "intervention": "PEEP_14> | PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_90 & PEEP_14>",
        },
        "PEEP_Intervention_Plan_FiO2_1.0": {
            # need to specify all higher values too because otherwise the expected values are incorrectly inferred
            "population": "COVID19 & VENTILATED & FiO2_100",
            "intervention": "PEEP_18>",
            "population_intervention": "COVID19 & VENTILATED & FiO2_100 & PEEP_18>",
        },
    }

    invalid_combinations = "(PEEP_18 & PEEP_14) | (PEEP_18 & PEEP_10) | (PEEP_18 & PEEP_8) | (PEEP_18 & PEEP_5) | (PEEP_14 & PEEP_10) | (PEEP_14 & PEEP_8) | (PEEP_14 & PEEP_5) | (PEEP_10 & PEEP_8) | (PEEP_10 & PEEP_5) | (PEEP_8 & PEEP_5)"

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        # NO_DATA FiO2 cols need to be set to "NEGATIVE", if any other FiO2 col is POSITIVE
        cols = [c for c in df.columns if c.startswith("FiO2_")]
        idx_any_positive = (df[cols] == IntervalType.POSITIVE).any(axis=1)

        for c in cols:
            idx_no_data = df[c] == IntervalType.NO_DATA
            df.loc[idx_any_positive & idx_no_data, c] = IntervalType.NEGATIVE

        return df

    def _insert_criteria_hook(self, person_entries, entry, row):
        if row["type"] == "measurement" and row["concept"].startswith("FiO2_"):
            # add another entry to make sure that FiO2 measurements are not valid longer than one hour,
            # as otherwise the determination of the expected data is very complex
            new_entry = copy.deepcopy(entry)
            new_entry.measurement_datetime = row["start_datetime"] + pd.Timedelta(
                minutes=59
            )
            new_entry.value_as_number = -100

            person_entries.append(new_entry)

    def test_recommendation_36a_peep(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
