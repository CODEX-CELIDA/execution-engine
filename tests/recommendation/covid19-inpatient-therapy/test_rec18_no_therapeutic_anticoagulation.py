import itertools

import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import (
    COVID19,
    AtrialFibrillation,
    CertoparinTherapeutic,
    Dalteparin100ie70kg2xd,
    Dalteparin200ie70kg1xd,
    Dalteparin10000ie50kg1xd,
    Dalteparin12500ie57kg1xd,
    Dalteparin15000ie82kg1xd,
    Dalteparin18000ie83kg1xd,
    Enoxaparin70kg1xd,
    Enoxaparin70kg2xd,
    Fondaparinux50_100kg1xd,
    Fondaparinux_lt50kg1xd,
    Fondaparinuxgt100kg1xd,
    IntensiveCare,
    Nadroparin50_59kg2xd,
    Nadroparin60_69kg2xd,
    Nadroparin70_79kg2xd,
    Nadroparin80_89kg2xd,
    Nadroparingt90kg2xd,
    Nadroparinlt50kg2xd,
    PulmonaryEmbolism,
    TinzaparinTherapeutic70kg1xd,
    VenousThrombosis,
    Weight40kg,
    Weight50kg,
    Weight57kg,
    Weight65kg,
    Weight70kg,
    Weight82kg,
    Weight83kg,
    Weight92kg,
    Weight110kg,
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
        "Anticoagulation_Plan_No_Specific_Indication": dict(
            population=COVID19()
            & IntensiveCare()
            & ~PulmonaryEmbolism()
            & ~VenousThrombosis()
            & ~AtrialFibrillation(),
            intervention=~(
                Dalteparin200ie70kg1xd
                | Dalteparin100ie70kg2xd
                | Dalteparin10000ie50kg1xd
                | Dalteparin12500ie57kg1xd
                | Dalteparin15000ie82kg1xd
                | Dalteparin18000ie83kg1xd
                | Enoxaparin70kg1xd
                | Enoxaparin70kg2xd
                | Nadroparinlt50kg2xd
                | Nadroparin50_59kg2xd
                | Nadroparin60_69kg2xd
                | Nadroparin70_79kg2xd
                | Nadroparin80_89kg2xd
                | Nadroparingt90kg2xd
                | CertoparinTherapeutic
                | TinzaparinTherapeutic70kg1xd
                | Fondaparinux_lt50kg1xd
                | Fondaparinux50_100kg1xd
                | Fondaparinuxgt100kg1xd
            ),
        ),
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
        Weight70kg & (rec18_population_combination | Enoxaparin70kg1xd),
        Weight70kg & (rec18_population_combination | Enoxaparin70kg2xd),
        Weight40kg & (rec18_population_combination | Nadroparinlt50kg2xd),
        Weight57kg & (rec18_population_combination | Nadroparin50_59kg2xd),
        Weight65kg & (rec18_population_combination | Nadroparin60_69kg2xd),
        Weight70kg & (rec18_population_combination | Nadroparin70_79kg2xd),
        Weight82kg & (rec18_population_combination | Nadroparin80_89kg2xd),
        Weight92kg & (rec18_population_combination | Nadroparingt90kg2xd),
        Weight70kg & (rec18_population_combination | CertoparinTherapeutic),
        Weight70kg & (rec18_population_combination | TinzaparinTherapeutic70kg1xd),
        Weight40kg & (rec18_population_combination | Fondaparinux_lt50kg1xd),
        Weight70kg & (rec18_population_combination | Fondaparinux50_100kg1xd),
        Weight110kg & (rec18_population_combination | Fondaparinuxgt100kg1xd),
        # Heparin (IV) and Argatroban are coded in the recommendation as "not any dose", thus we can't test them here.
        # rec18_population_combination | HeparinIV100ie | APTT(comparator="<"),
        # rec18_population_combination | Argatroban100mg | APTT(comparator="<"),
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        Weight70kg
        & (
            rec18_population_combination
            | Dalteparin200ie70kg1xd
            | Enoxaparin70kg2xd
            | Nadroparin70_79kg2xd
            | TinzaparinTherapeutic70kg1xd
        ),
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
                "?DALTEPARIN",
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
