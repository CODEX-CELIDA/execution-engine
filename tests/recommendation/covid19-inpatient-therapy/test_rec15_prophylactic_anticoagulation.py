import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util.types import TimeRange
from tests.recommendation.test_recommendation_base import TestRecommendationBase


class TestRecommendation15ProphylacticAnticoagulation_v1_3(TestRecommendationBase):
    """
    Test the recommendation for prophylactic anticoagulation.

    This class tests the v1.3 version of the recommendation 15 that only checks for the existence
    of one or more prophylactic drugs.
    """

    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
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
            "AntithromboticProphylaxisWithLWMH": {
                "population": "COVID19 & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
                "intervention": "DALTEPARIN= | ENOXAPARIN= | NADROPARIN_LOW_WEIGHT= | NADROPARIN_HIGH_WEIGHT= | CERTOPARIN= | TINZAPARIN= ",
            },
            "AntithromboticProphylaxisWithFondaparinux": {
                "population": "COVID19 & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
                "intervention": "FONDAPARINUX_PROPHYLACTIC=",
            },
        }

    @pytest.fixture
    def invalid_combinations(self, population_intervention: dict) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_15_prophylactic_anticoagulation(
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


class TestRecommendation15ProphylacticAnticoagulation_v1_2(TestRecommendationBase):
    """
    Test the recommendation for prophylactic anticoagulation with thrombosis.

    This class tests the v1.2 version of the recommendation 15 that includes thrombosis as a
    population criterion.
    """

    @pytest.fixture
    def recommendation_url(self) -> str:
        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
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
            "AntithromboticProphylaxisWithLWMH": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
                "intervention": "DALTEPARIN= | ENOXAPARIN= | NADROPARIN_LOW_WEIGHT= | NADROPARIN_HIGH_WEIGHT= | CERTOPARIN= | TINZAPARIN= ",
            },
            "AntithromboticProphylaxisWithFondaparinux": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
                "intervention": "FONDAPARINUX_PROPHYLACTIC=",
            },
            "NoAntithromboticProphylaxis": {
                "population": "COVID19 & VENOUS_THROMBOSIS",
                "intervention": "~(DALTEPARIN= | ENOXAPARIN= | NADROPARIN_LOW_WEIGHT= | NADROPARIN_HIGH_WEIGHT= | CERTOPARIN= | TINZAPARIN= | FONDAPARINUX_PROPHYLACTIC=)",
            },
        }

    @pytest.fixture
    def invalid_combinations(self, population_intervention: dict) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_15_prophylactic_anticoagulation(
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
