import itertools

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import *
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

rec15_population_no_contraindications = (
    COVID19() | ~HIT2() | ~HeparinAllergy() | ~HeparinoidAllergy()
)

rec15_population_with_contraindications = (
    COVID19() & HIT2() | HeparinAllergy() | HeparinoidAllergy()
)


class TestRecommendation15ProphylacticAnticoagulation_v1_5(TestRecommendationBaseV2):
    """
    Test the recommendation for prophylactic anticoagulation.

    This class tests the v1.5 version of the recommendation 15.
    """

    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
    )
    recommendation_package_version = "v1.5.0-snapshot"

    recommendation_expression = {
        "AntithromboticProphylaxisWithLWMH": dict(
            population=COVID19() & ~HIT2() & ~HeparinAllergy() & ~HeparinoidAllergy(),
            intervention=(
                (
                    Dalteparin2500ie1xd
                    | Dalteparin5000ie1xd
                    | NadroparinProphylactic3800ie1xd
                    | NadroparinProphylactic5700ie1xd
                    | Certoparin3000ie1xd
                    | Tinzaparin3500ie1xd
                    | HeparinSubcutaneous5000ie2xd
                    | HeparinSubcutaneous5000ie3xd
                    | HeparinSubcutaneous75002xd
                )
                & ~(
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
                )
            ),
        ),
        "AntithromboticProphylaxisWithFondaparinux": dict(
            population=COVID19() & (HIT2() | HeparinAllergy() | HeparinoidAllergy()),
            intervention=Fondaparinux2_5mg1xd
            & ~(
                Fondaparinux_lt50kg1xd
                | Fondaparinux50_100kg1xd
                | Fondaparinuxgt100kg1xd
            ),
        ),
    }

    combinations = [
        # prophylactic anticoagulations (without contraindications)
        rec15_population_no_contraindications | Dalteparin2500ie1xd,
        rec15_population_no_contraindications | Dalteparin5000ie1xd,
        Weight70kg
        & (rec15_population_no_contraindications | NadroparinProphylactic3800ie1xd),
        Weight82kg
        & (rec15_population_no_contraindications | NadroparinProphylactic5700ie1xd),
        rec15_population_no_contraindications | Certoparin3000ie1xd,
        rec15_population_no_contraindications | Tinzaparin3500ie1xd,
        rec15_population_no_contraindications | HeparinSubcutaneous5000ie2xd,
        rec15_population_no_contraindications | HeparinSubcutaneous5000ie3xd,
        rec15_population_no_contraindications | HeparinSubcutaneous75002xd,
        # prophylactic anticoagulations (with contraindications)
        rec15_population_with_contraindications | Fondaparinux2_5mg1xd,
        # therapeutic anticoagulations (without contraindications)
        Weight70kg & (rec15_population_no_contraindications | ~Dalteparin200ie70kg1xd),
        Weight70kg & (rec15_population_no_contraindications | ~Dalteparin100ie70kg2xd),
        Weight50kg
        & (rec15_population_no_contraindications | ~Dalteparin10000ie50kg1xd),
        Weight57kg
        & (rec15_population_no_contraindications | ~Dalteparin12500ie57kg1xd),
        Weight82kg
        & (rec15_population_no_contraindications | ~Dalteparin15000ie82kg1xd),
        Weight83kg
        & (rec15_population_no_contraindications | ~Dalteparin18000ie83kg1xd),
        Weight70kg & (rec15_population_no_contraindications | ~Enoxaparin70kg1xd),
        Weight70kg & (rec15_population_no_contraindications | ~Enoxaparin70kg2xd),
        Weight40kg & (rec15_population_no_contraindications | ~Nadroparinlt50kg2xd),
        Weight57kg & (rec15_population_no_contraindications | ~Nadroparin50_59kg2xd),
        Weight65kg & (rec15_population_no_contraindications | ~Nadroparin60_69kg2xd),
        Weight70kg & (rec15_population_no_contraindications | ~Nadroparin70_79kg2xd),
        Weight82kg & (rec15_population_no_contraindications | ~Nadroparin80_89kg2xd),
        Weight92kg & (rec15_population_no_contraindications | ~Nadroparingt90kg2xd),
        Weight70kg & (rec15_population_no_contraindications | ~CertoparinTherapeutic),
        Weight70kg
        & (rec15_population_no_contraindications | ~TinzaparinTherapeutic70kg1xd),
        # therapeutic anticoagulations (with contraindications)
        Weight40kg
        & (rec15_population_with_contraindications | ~Fondaparinux_lt50kg1xd),
        Weight70kg
        & (rec15_population_with_contraindications | ~Fondaparinux50_100kg1xd),
        Weight110kg
        & (rec15_population_with_contraindications | ~Fondaparinuxgt100kg1xd),
        # prophylactic + therapeutic anticoagulations (without contraindications)
        Weight82kg
        & (
            rec15_population_no_contraindications
            | Dalteparin2500ie1xd
            | Nadroparin80_89kg2xd
        ),
        Weight70kg
        & (
            rec15_population_no_contraindications
            | HeparinSubcutaneous5000ie3xd
            | Enoxaparin70kg2xd
        ),
    ]

    def test_recommendation_15_prophylactic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


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

    combinations = (
        [
            # Population: All combinations
            # Intervention: All prophylactic anticoagulations criteria (each optional)
            " ".join(pair)
            for pair in itertools.product(
                ["?COVID19 ?HIT2 ?HEPARIN_ALLERGY ?HEPARINOID_ALLERGY"],
                [
                    "?DALTEPARIN=",
                    "?ENOXAPARIN=",
                    "?NADROPARIN_LOW_WEIGHT=",
                    "?NADROPARIN_HIGH_WEIGHT=",
                    "?CERTOPARIN=",
                    "?TINZAPARIN=",
                    "?HEPARIN_SUBCUTANEOUS=",
                    "?FONDAPARINUX_PROPHYLACTIC=",
                ],
            )
        ]
        + [
            # Population: The inclusion and one exclusion criterion (all combinations)
            # Intervention: All therapeutic anticoagulation criteria (each optional)
            " ".join(pair)
            for pair in itertools.product(
                ["?COVID19 ?HIT2"],
                [
                    "?DALTEPARIN>",
                    "?ENOXAPARIN>",
                    "?NADROPARIN_LOW_WEIGHT>",
                    "?NADROPARIN_HIGH_WEIGHT>",
                    "?CERTOPARIN>",
                    "?HEPARIN=",
                    "?ARGATROBAN=",
                ],
            )
        ]
        + [
            # Population: The inclusion criterion
            # Intervention: One prophylatic criterion (always) + all therapeutic anticoagulation (each optional)
            " ".join(pair)
            for pair in itertools.product(
                ["COVID19 ENOXAPARIN="],
                [
                    "?DALTEPARIN>",
                    "?ENOXAPARIN>",
                    "?NADROPARIN_LOW_WEIGHT>",
                    "?NADROPARIN_HIGH_WEIGHT>",
                    "?CERTOPARIN>",
                    "?HEPARIN=",
                    "?ARGATROBAN=",
                ],
            )
        ]
    )

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
