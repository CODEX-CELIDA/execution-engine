import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util import TimeRange
from tests.recommendation.test_recommendation import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation36aPeep(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-ards-peep"

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.3": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_30",
                # need to specify all higher values too because otherwise the expected values are incorrectly inferred
                "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.4": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_40",
                "intervention": "PEEP_5> | PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.5": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_50",
                "intervention": "PEEP_8> | PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.6": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_60",
                "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.7": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_70",
                "intervention": "PEEP_10> | PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.8": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_80",
                "intervention": "PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_0.9": {
                "population": "COVID19 & VENTILATED & ARDS & FiO2_90",
                "intervention": "PEEP_14> | PEEP_18>",
            },
            "PEEP_Intervention_Plan_With_ARDS_FiO2_1.0": {
                # need to specify all higher values too because otherwise the expected values are incorrectly inferred
                "population": "COVID19 & VENTILATED & ARDS & FiO2_100",
                "intervention": "PEEP_18>",
            },
        }

    # @pytest.fixture
    # def invalid_combinations(self):

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
