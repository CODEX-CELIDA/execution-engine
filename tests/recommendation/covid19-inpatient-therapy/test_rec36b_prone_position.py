import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation36bPronePosition(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards"
    )
    recommendation_package_version = "latest"
    recommendation_expression = {
        "Abdominal_Positioning_Plan_In_Case_Of_ARDS_And_PaO2/FiO2_<_150mmHg": {
            "population": "COVID19 & VENTILATED & ARDS & OXYGENATION_INDEX<",
            "intervention": "PRONING>",
        },
    }

    def test_recommendation_36b_proning(self, setup_testdata) -> None:
        self.recommendation_test_runner(setup_testdata)
