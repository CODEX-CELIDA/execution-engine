import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import TimeRange
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation36aPeep(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-peep"
        )

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "PEEP_Intervention_Plan_FiO2_0.3": {
                "population": "COVID19 & VENTILATED & FiO2_30",
                # need to specify all higher values too because otherwise the expected values are incorrectly inferred
                "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.4": {
                "population": "COVID19 & VENTILATED & FiO2_40",
                "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.5": {
                "population": "COVID19 & VENTILATED & FiO2_50",
                "intervention": "PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.6": {
                "population": "COVID19 & VENTILATED & FiO2_60",
                "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.7": {
                "population": "COVID19 & VENTILATED & FiO2_70",
                "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.8": {
                "population": "COVID19 & VENTILATED & FiO2_80",
                "intervention": "PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_0.9": {
                "population": "COVID19 & VENTILATED & FiO2_90",
                "intervention": "PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_FiO2_1.0": {
                # need to specify all higher values too because otherwise the expected values are incorrectly inferred
                "population": "COVID19 & VENTILATED & FiO2_100",
                "intervention": "PEEP_18>",
            },
        }

    @pytest.fixture
    def invalid_combinations(self) -> str:
        return "(PEEP_18 & PEEP_14) | (PEEP_18 & PEEP_10) | (PEEP_18 & PEEP_8) | (PEEP_18 & PEEP_5) | (PEEP_14 & PEEP_10) | (PEEP_14 & PEEP_8) | (PEEP_14 & PEEP_5) | (PEEP_10 & PEEP_8) | (PEEP_10 & PEEP_5) | (PEEP_8 & PEEP_5)"

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        # NO_DATA FiO2 cols need to be set to "NEGATIVE", if any other FiO2 col is POSITIVE
        cols = [c for c in df.columns if c.startswith("FiO2_")]
        idx_any_positive = (df[cols] == IntervalType.POSITIVE).any(axis=1)

        for c in cols:
            idx_no_data = df[c] == IntervalType.NO_DATA
            df.loc[idx_any_positive & idx_no_data, c] = IntervalType.NEGATIVE

        return df

    def test_recommendation_36a_peep(
        self,
        criteria_extended: pd.DataFrame,
        observation_window: TimeRange,
        recommendation_url: str,
    ) -> None:
        self.recommendation_test_runner(
            recommendation_url=recommendation_url,
            observation_window=observation_window,
            criteria_extended=criteria_extended,
        )
