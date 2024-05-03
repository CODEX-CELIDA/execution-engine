import itertools

import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import (
    COVID19,
    AtrialFibrillation,
    Dalteparin200ie70kg1xd,
    Dalteparin200ie70kg2xd,
    IntensiveCare,
    PulmonaryEmbolism,
    VenousThrombosis,
    Weight70kg,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2


@pytest.mark.recommendation
class TestRecommendation18NoTherapeuticAnticoagulation_v1_5(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.5.0-snapshot"

    # TODO:
    #   - NOT operator for OrGenerator doesn't seem to exist (see "intervention" below)
    #   - Is it correct to list Dalteparin200ie70kg1xd and Dalteparin200ie70kg2xd here? Are these the actual criteria?
    #       They are defined for a specific weight - is this what I intend to do here?
    #   - In general, all drug criteria should be defined only once, as they are the same for all recommendations (I think)
    recommendation_expression = {
        "Anticoagulation_Plan_No_Specific_Indication": {
            "population": COVID19()
            & IntensiveCare()
            & ~PulmonaryEmbolism()
            & ~VenousThrombosis()
            & ~AtrialFibrillation(),
            "intervention": ~(
                Dalteparin200ie70kg1xd | Dalteparin200ie70kg2xd
            ),  # | Enoxaparin() | NadroparinLowWeight() | NadroparinHighWeight() | Certoparin() | (Heparin() & APTT()) | (Argatroban() & APTT())),
        },
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        Weight70kg
        & (
            COVID19()
            | IntensiveCare()
            | ~PulmonaryEmbolism()
            | ~VenousThrombosis()
            | ~AtrialFibrillation()
            | Dalteparin200ie70kg1xd
        ),
        Weight70kg
        & (
            COVID19()
            | IntensiveCare()
            | ~PulmonaryEmbolism()
            | ~VenousThrombosis()
            | ~AtrialFibrillation()
            | Dalteparin200ie70kg2xd
        ),
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
