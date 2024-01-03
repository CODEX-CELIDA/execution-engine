import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.types import TimeRange
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation36bPronePosition(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards"

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "Abdominal_Positioning_Plan_In_Case_Of_ARDS_And_PaO2/FiO2_<_150mmHg": {
                "population": "COVID19 & VENTILATED & ARDS & OXYGENATION_INDEX<",
                "intervention": "PRONING>",
            },
        }

    def test_recommendation_36b_proning(
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
