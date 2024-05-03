import itertools

import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import (
    COVID19,
    AtrialFibrillation,
    Dalteparin100ie70kg2xd,
    Dalteparin200ie70kg1xd,
    Dalteparin10000ie50kg1xd,
    Dalteparin12500ie57kg1xd,
    Dalteparin15000ie82kg1xd,
    Dalteparin18000ie83kg1xd,
    IntensiveCare,
    PulmonaryEmbolism,
    VenousThrombosis,
    Weight50kg,
    Weight57kg,
    Weight70kg,
    Weight82kg,
    Weight83kg,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

rec18_population_combination = (
    COVID19()
    | IntensiveCare()
    | ~PulmonaryEmbolism()
    | ~VenousThrombosis()
    | ~AtrialFibrillation()
)


@pytest.mark.recommendation
class TestRecommendation18NoTherapeuticAnticoagulation_v1_5(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.5.0-snapshot"

    recommendation_expression = {
        "Anticoagulation_Plan_No_Specific_Indication": {
            "population": COVID19()
            & IntensiveCare()
            & ~PulmonaryEmbolism()
            & ~VenousThrombosis()
            & ~AtrialFibrillation(),
            "intervention": ~(
                Dalteparin200ie70kg1xd
                | Dalteparin100ie70kg2xd
                | Dalteparin10000ie50kg1xd
                | Dalteparin12500ie57kg1xd
                | Dalteparin15000ie82kg1xd
                | Dalteparin18000ie83kg1xd
            ),  # | Enoxaparin() | NadroparinLowWeight() | NadroparinHighWeight() | Certoparin() | (Heparin() & APTT()) | (Argatroban() & APTT())),
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        Weight70kg & (rec18_population_combination | Dalteparin200ie70kg1xd),
        Weight70kg & (rec18_population_combination | Dalteparin100ie70kg2xd),
        Weight50kg & (rec18_population_combination | Dalteparin10000ie50kg1xd),
        Weight57kg & (rec18_population_combination | Dalteparin12500ie57kg1xd),
        Weight82kg & (rec18_population_combination | Dalteparin15000ie82kg1xd),
        Weight83kg & (rec18_population_combination | Dalteparin18000ie83kg1xd),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Enoxaparin(),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | NadroparinLowWeight(),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | NadroparinHighWeight(),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Certoparin(),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Heparin() | APTT(),
        # COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Argatroban() | APTT(),
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        #     COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Heparin() | APTT() | Argatroban(),
        #     (COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation()) & Dalteparin() & NadroparinLowWeight(),
        #     (COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation()) & NadroparinHighWeight() & Enoxaparin(),
        #     COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation() | Certoparin() | Heparin() | APTT(),
        #     (COVID19() | IntensiveCare() | ~PulmonaryEmbolism() | ~VenousThrombosis() | ~AtrialFibrillation()) & NadroparinHighWeight() & Argatroban() & APTT(),
    ]

    # invalid_combinations = NadroparinHighWeight() & NadroparinLowWeight()

    def test_recommendation_18_no_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


@pytest.mark.recommendation
class TestRecommendation18NoTherapeuticAnticoagulation_v1_4(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.4.0"

    recommendation_expression = {
        "Anticoagulation_Plan_No_Specific_Indication": {
            "population": "COVID19 & ICU & ~PULMONARY_EMBOLISM & ~VENOUS_THROMBOSIS & ~ATRIAL_FIBRILLATION",
            "intervention": "~(DALTEPARIN> | ENOXAPARIN> | NADROPARIN_LOW_WEIGHT> | NADROPARIN_HIGH_WEIGHT> | CERTOPARIN> | (HEPARIN & APTT>) | (ARGATROBAN & APTT>))",
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        " ".join(pair)
        for pair in itertools.product(
            [
                "?COVID19 ?ICU ?PULMONARY_EMBOLISM ?VENOUS_THROMBOSIS ?ATRIAL_FIBRILLATION"
            ],
            [
                "?Dalteparin()",
                "?ENOXAPARIN>",
                "?NADROPARIN_LOW_WEIGHT>",
                "?NADROPARIN_HIGH_WEIGHT>",
                "?CERTOPARIN>",
                "?HEPARIN ?APTT>",
                "?ARGATROBAN ?APTT>",
            ],
        )
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        " ".join(pair)
        for pair in itertools.product(
            [
                "?COVID19 ?ICU ?PULMONARY_EMBOLISM ?VENOUS_THROMBOSIS ?ATRIAL_FIBRILLATION"
            ],
            [
                "?HEPARIN ?APTT> ?ARGATROBAN",
                "DALTEPARIN> NADROPARIN_LOW_WEIGHT>",
                "NADROPARIN_HIGH_WEIGHT> ENOXAPARIN>",
                "CERTOPARIN> ?HEPARIN ?APTT>",
                "NADROPARIN_HIGH_WEIGHT> ARGATROBAN ?APTT>",
            ],
        )
    ]

    invalid_combinations = "NADROPARIN_HIGH_WEIGHT> & NADROPARIN_LOW_WEIGHT>"

    def test_recommendation_18_no_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)
