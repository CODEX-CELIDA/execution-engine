from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import (
    ARDS,
    HeightByIdealBodyWeight,
    TidalVolume,
    Ventilated,
)
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

HeightFor70kgIdealBodyWeight = HeightByIdealBodyWeight(ideal_body_weight=70)


class TestRecommendationSepsisM1TidalVolume(TestRecommendationBaseV2):
    recommendation_url = "sepsis/recommendation/ventilation-plan-ards-tidal-volume"
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Ventilation_Plan_ARDS": {
            "population": Ventilated() & ARDS(),
            "intervention": TidalVolume(
                weight=HeightFor70kgIdealBodyWeight.ideal_body_weight
            )
            & HeightFor70kgIdealBodyWeight,
        },
    }

    combinations = [
        Ventilated()
        | HeightFor70kgIdealBodyWeight
        | ARDS()
        | TidalVolume(weight=HeightFor70kgIdealBodyWeight.ideal_body_weight)
    ]

    def test_recommendation_m1_tidal_volume(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
