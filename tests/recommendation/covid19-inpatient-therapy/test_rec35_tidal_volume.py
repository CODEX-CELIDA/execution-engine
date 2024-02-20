import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation35TidalVolume(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume"
    )
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Ventilation_Plan": {
            "population": "COVID19 & VENTILATED & ARDS",
            "intervention": "TIDAL_VOLUME< & PPLATEAU<",
        },
    }

    def test_recommendation_35_tidal_volume(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
