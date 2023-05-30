import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util import TimeRange
from tests.recommendation.test_recommendation import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendationSepsisM1TidalVolume(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = "sepsis/recommendation/ventilation-plan-ards-tidal-volume"

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "Ventilation_Plan_ARDS": {
                "population": "VENTILATED & ARDS",
                "intervention": "TIDAL_VOLUME<",
            },
        }

    def test_recommendation_m1_tidal_volume(
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
