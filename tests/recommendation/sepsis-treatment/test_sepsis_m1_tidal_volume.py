import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendationSepsisM1TidalVolume(TestRecommendationBase):
    recommendation_url = "sepsis/recommendation/ventilation-plan-ards-tidal-volume"
    recommendation_package_version = "latest"
    recommendation_expression = {
        "Ventilation_Plan_ARDS": {
            "population": "VENTILATED & ARDS",
            "intervention": "TIDAL_VOLUME<",
        },
    }

    def test_recommendation_m1_tidal_volume(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
