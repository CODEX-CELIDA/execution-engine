from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


class TestRecommendation15ProphylacticAnticoagulation_v1_4(TestRecommendationBase):
    """
    Test the recommendation for prophylactic anticoagulation.

    This class tests the v1.4 version of the recommendation 15.
    """

    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    )
    recommendation_package_version = "v1.4.0-snapshot"

    recommendation_expression = {
        "AntithromboticProphylaxisWithLWMH": {
            "population": "COVID19 & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
            "intervention": "Eq("
            "DALTEPARIN= + ENOXAPARIN= + NADROPARIN_LOW_WEIGHT= + NADROPARIN_HIGH_WEIGHT= + CERTOPARIN= + TINZAPARIN= + HEPARIN_SUBCUTANEOUS=, "
            "1) & "
            "~(DALTEPARIN> | ENOXAPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | CERTOPARIN> | HEPARIN= | ARGATROBAN=)",
        },
        "AntithromboticProphylaxisWithFondaparinux": {
            "population": "COVID19 & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
            "intervention": "FONDAPARINUX_PROPHYLACTIC=",
        },
    }

    invalid_combinations = "(NADROPARIN_HIGH_WEIGHT> & NADROPARIN_LOW_WEIGHT>) | (NADROPARIN_HIGH_WEIGHT= & NADROPARIN_LOW_WEIGHT=)"

    def test_recommendation_15_prophylactic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


class TestRecommendation15ProphylacticAnticoagulation_v1_3(TestRecommendationBase):
    """
    Test the recommendation for prophylactic anticoagulation.

    This class tests the v1.3 version of the recommendation 15 that only checks for the existence
    of one or more prophylactic drugs.
    """

    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    )
    recommendation_package_version = "v1.3.1"

    recommendation_expression = {
        "AntithromboticProphylaxisWithLWMH": {
            "population": "COVID19 & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
            "intervention": "DALTEPARIN= | ENOXAPARIN= | NADROPARIN_LOW_WEIGHT= | NADROPARIN_HIGH_WEIGHT= | CERTOPARIN= | TINZAPARIN= ",
        },
        "AntithromboticProphylaxisWithFondaparinux": {
            "population": "COVID19 & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY)",
            "intervention": "FONDAPARINUX_PROPHYLACTIC=",
        },
    }

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_15_prophylactic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


class TestRecommendation15ProphylacticAnticoagulation_v1_2(TestRecommendationBase):
    """
    Test the recommendation for prophylactic anticoagulation with thrombosis.

    This class tests the v1.2 version of the recommendation 15 that includes thrombosis as a
    population criterion.
    """

    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    )
    recommendation_package_version = "v1.2.2"

    recommendation_expression = {
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

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_15_prophylactic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)
