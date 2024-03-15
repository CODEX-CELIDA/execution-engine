import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.value import ValueNumber
from tests._fixtures import concept
from tests._testdata.generator.generators import (
    ARDS,
    COVID19,
    PPlateau,
    TidalVolume,
    Ventilated,
    Weight,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

Weight70kg = Weight(ValueNumber(value=70, unit=concept.concept_unit_kg), comparator=">")


class TestRecommendation35TidalVolumeV2(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume"
    )
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Ventilation_Plan": {
            "population": COVID19() & Ventilated() & ARDS(),
            "intervention": TidalVolume(weight=Weight70kg) & PPlateau(comparator="<"),
        },
    }

    combinations = [
        COVID19()
        | Ventilated()
        | ARDS()
        | TidalVolume(weight=Weight70kg)
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
