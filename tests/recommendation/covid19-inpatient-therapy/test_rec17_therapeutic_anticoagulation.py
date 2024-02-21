import itertools

import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
    )
    recommendation_package_version = "latest"

    recommendation_expression = {
        "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": {
            "population": "COVID19 & ~ICU & D_DIMER",
            "intervention": "(HEPARIN & APTT>) | DALTEPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | ENOXAPARIN> | CERTOPARIN> | FONDAPARINUX_THERAPEUTIC= | (ARGATROBAN & APTT>)",
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        " ".join(pair)
        for pair in itertools.product(
            ["?COVID19 ?ICU ?D_DIMER"],
            [
                "?HEPARIN ?APTT>",
                "?DALTEPARIN>",
                "?NADROPARIN_LOW_WEIGHT>",
                "?NADROPARIN_HIGH_WEIGHT>",
                "?ENOXAPARIN>",
                "?CERTOPARIN>",
                "?FONDAPARINUX_THERAPEUTIC=",
                "?ARGATROBAN ?APTT>",
            ],
        )
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        " ".join(pair)
        for pair in itertools.product(
            ["?COVID19 ?ICU ?D_DIMER"],
            [
                "?HEPARIN ?APTT> ?ARGATROBAN",
                "DALTEPARIN> NADROPARIN_LOW_WEIGHT>",
                "NADROPARIN_HIGH_WEIGHT> ENOXAPARIN>",
                "CERTOPARIN> FONDAPARINUX_THERAPEUTIC=",
                "FONDAPARINUX_THERAPEUTIC= ARGATROBAN ?APTT>",
            ],
        )
    ]

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT> & NADROPARIN_LOW_WEIGHT>"

    def test_recommendation_17_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation_v1_2(TestRecommendationBase):
    """
    Test the recommendation for therapeutic anticoagulation with thrombosis.

    This class tests the "old" version of the recommendation 17 (<= 1.2) that includes thrombosis as a
    population criterion.
    """

    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.2.2"

    recommendation_expression = {
        "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": {
            "population": "COVID19 & ~ICU & ~VENOUS_THROMBOSIS & D_DIMER",
            "intervention": "(HEPARIN & APTT>) | DALTEPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | ENOXAPARIN> | CERTOPARIN> | FONDAPARINUX_THERAPEUTIC= | (ARGATROBAN & APTT>)",
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        " ".join(pair)
        for pair in itertools.product(
            ["?COVID19 ?ICU ?VENOUS_THROMBOSIS ?D_DIMER"],
            [
                "?HEPARIN ?APTT>",
                "?DALTEPARIN>",
                "?NADROPARIN_LOW_WEIGHT>",
                "?NADROPARIN_HIGH_WEIGHT>",
                "?ENOXAPARIN>",
                "?CERTOPARIN>",
                "?FONDAPARINUX_THERAPEUTIC=",
                "?ARGATROBAN ?APTT>",
            ],
        )
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        " ".join(pair)
        for pair in itertools.product(
            ["?COVID19 ?ICU ?VENOUS_THROMBOSIS ?D_DIMER"],
            [
                "?HEPARIN ?APTT> ?ARGATROBAN",
                "DALTEPARIN> NADROPARIN_LOW_WEIGHT>",
                "NADROPARIN_HIGH_WEIGHT> ENOXAPARIN>",
                "CERTOPARIN> FONDAPARINUX_THERAPEUTIC=",
                "FONDAPARINUX_THERAPEUTIC= ARGATROBAN ?APTT>",
            ],
        )
    ]

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT> & NADROPARIN_LOW_WEIGHT>"

    def test_recommendation_17_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)
