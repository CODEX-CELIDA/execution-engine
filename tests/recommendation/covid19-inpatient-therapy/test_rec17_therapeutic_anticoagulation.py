import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.types import TimeRange
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation(TestRecommendationBase):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
        )

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def recommendation_package_version(self) -> str:
        """
        Version of the recommendation FHIR package

        Required to allow different versions of the recommendation package to be tested.
        """
        return "v1.3.1"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": {
                "population": "COVID19 & ~ICU & D_DIMER",
                "intervention": "(HEPARIN & APTT>) | DALTEPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | ENOXAPARIN> | CERTOPARIN> | FONDAPARINUX_THERAPEUTIC= | (ARGATROBAN & APTT>)",
            },
        }

    @pytest.fixture
    def invalid_combinations(self) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_17_therapeutic_anticoagulation(
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


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulationOldVersionWithThrombosis(
    TestRecommendationBase
):
    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
        )

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def recommendation_package_version(self) -> str:
        """
        Version of the recommendation FHIR package

        Required to allow different versions of the recommendation package to be tested.
        """
        return "v1.2.2"

    @pytest.fixture
    def population_intervention(self) -> dict:
        return {
            "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": {
                "population": "COVID19 & ~ICU & ~VENOUS_THROMBOSIS & D_DIMER",
                "intervention": "(HEPARIN & APTT>) | DALTEPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | ENOXAPARIN> | CERTOPARIN> | FONDAPARINUX_THERAPEUTIC= | (ARGATROBAN & APTT>)",
            },
        }

    @pytest.fixture
    def invalid_combinations(self) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_17_therapeutic_anticoagulation(
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
