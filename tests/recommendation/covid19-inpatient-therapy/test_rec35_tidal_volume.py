import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.types import TimeRange
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation35TidalVolume(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume"

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def recommendation_package_version(self) -> str:
        """
        Version of the recommendation FHIR package

        Required to allow different versions of the recommendation package to be tested.
        """
        return "latest"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "Ventilation_Plan": {
                "population": "COVID19 & VENTILATED & ARDS",
                "intervention": "TIDAL_VOLUME< & PPLATEAU<",
            },
        }

    def test_recommendation_35_tidal_volume(
        self,
        criteria_extended: pd.DataFrame,
        observation_window: TimeRange,
        recommendation_url: str,
        recommendation_package_version: str,
    ) -> None:
        self.recommendation_test_runner(
            recommendation_url=recommendation_url,
            observation_window=observation_window,
            criteria_extended=criteria_extended,
            recommendation_package_version=recommendation_package_version,
        )
