import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import (
    ARDS,
    COVID19,
    HeightByIdealBodyWeight,
    PPlateau,
    TidalVolume,
    Ventilated,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

HeightFor70kgIdealBodyWeight = HeightByIdealBodyWeight(ideal_body_weight=70)


class TestRecommendation35TidalVolumeV2(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume"
    )
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Ventilation_Plan": {
            "population": COVID19() & Ventilated() & ARDS(),
            "intervention": TidalVolume(
                weight=HeightFor70kgIdealBodyWeight.ideal_body_weight
            )
            & PPlateau(comparator="<")
            & HeightFor70kgIdealBodyWeight,
        },
    }

    combinations = [
        COVID19()
        | Ventilated()
        | HeightFor70kgIdealBodyWeight
        | ARDS()
        | TidalVolume(weight=HeightFor70kgIdealBodyWeight.ideal_body_weight)
        | PPlateau(comparator="<"),
    ]

    def test_recommendation_35_tidal_volume(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)


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
