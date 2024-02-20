import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation18NoTherapeuticAnticoagulation(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation"
    )
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Anticoagulation_Plan_No_Specific_Indication": {
            "population": "COVID19 & ICU & ~PULMONARY_EMBOLISM & ~VENOUS_THROMBOSIS & ~ATRIAL_FIBRILLATION",
            "intervention": "~(DALTEPARIN> | ENOXAPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | CERTOPARIN> | (HEPARIN & APTT>) | (ARGATROBAN & APTT>))",
        },
    }

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_18_no_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)
