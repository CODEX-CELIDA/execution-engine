import itertools

import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests._testdata.generator.generators import *
from tests.recommendation.covid19_inpatient_therapy.test_rec18_no_therapeutic_anticoagulation import (
    intervention_therapeutic_anticoagulation,
)
from tests.recommendation.test_recommendation_base import TestRecommendationBase
from tests.recommendation.test_recommendation_base_v2 import TestRecommendationBaseV2

rec17_population_combination = COVID19() | IntensiveCare() | DDimer()


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation_v1_5(TestRecommendationBaseV2):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.5.0-snapshot"
    recommendation_parser_version = 2  # for recommendation package version >= 1.4

    recommendation_expression = {
        "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": dict(
            population=COVID19() & ~IntensiveCare() & DDimer(),
            intervention=intervention_therapeutic_anticoagulation,
        )
    }

    combinations = [
        # Population: All combinations
        # Intervention: All therapeutic anticoagulation criteria (each optional)
        Weight70kg & (rec17_population_combination | Dalteparin200ie70kg1xd),
        Weight70kg & (rec17_population_combination | Dalteparin100ie70kg2xd),
        Weight50kg & (rec17_population_combination | Dalteparin10000ie50kg1xd),
        Weight57kg & (rec17_population_combination | Dalteparin12500ie57kg1xd),
        Weight82kg & (rec17_population_combination | Dalteparin15000ie82kg1xd),
        Weight83kg & (rec17_population_combination | Dalteparin18000ie83kg1xd),
        Weight70kg & (rec17_population_combination | Enoxaparin70kg1xd),
        Weight70kg & (rec17_population_combination | Enoxaparin70kg2xd),
        Weight40kg & (rec17_population_combination | Nadroparinlt50kg2xd),
        Weight57kg & (rec17_population_combination | Nadroparin50_59kg2xd),
        Weight65kg & (rec17_population_combination | Nadroparin60_69kg2xd),
        Weight70kg & (rec17_population_combination | Nadroparin70_79kg2xd),
        Weight82kg & (rec17_population_combination | Nadroparin80_89kg2xd),
        Weight92kg & (rec17_population_combination | Nadroparingt90kg2xd),
        Weight70kg & (rec17_population_combination | CertoparinTherapeutic),
        Weight70kg & (rec17_population_combination | TinzaparinTherapeutic70kg1xd),
        Weight40kg & (rec17_population_combination | Fondaparinux_lt50kg1xd),
        Weight70kg & (rec17_population_combination | Fondaparinux50_100kg1xd),
        Weight110kg & (rec17_population_combination | Fondaparinuxgt100kg1xd),
        rec17_population_combination | HeparinIV100ie | APTT(comparator=">"),
        rec17_population_combination | Argatroban100mg | APTT(comparator=">"),
    ] + [
        # Population: All combinations
        # Intervention: Some double combinations
        Weight70kg
        & (
            rec17_population_combination
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

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The recommendation uses ~(ICU | VENOUS_THROMBOSIS) as a population criterion, and at least
        ICU ends within a day, i.e. the day has both POSITIVE and NEGATIVE intervals for ICU.
        Now, for population, POSITIVE takes precedence over NEGATIVE, so the day is marked as POSITIVE.
        However, the expected value for the day is determined on basis of the whole day, and inverting the POSITIVE
        label gets NEGATIVE (as a label for the whole day), although there is actually both NEGATIVE and POSITIVE during
        the day, so inversion would lead to still POSITIVE and NEGATIVE being there, and so the day would be POSITIVE,
        if we would consider within-day granularity, which, unfortunately, we do not (here, in testing!).

        So we just forcibly set the last entry of POSITIVE to NEGATIVE.
        """
        cols = ["INTENSIVE_CARE"]

        def set_last_positive_to_negative(group):
            idx_positive = group[group == IntervalType.POSITIVE].index
            if not idx_positive.empty:
                last_positive_idx = idx_positive[-1]
                group.at[last_positive_idx] = IntervalType.NEGATIVE
            return group

        for c in cols:
            df[c] = (
                df[c]
                .groupby(level="person_id", group_keys=False)
                .apply(set_last_positive_to_negative)
            )

        return df

    def test_recommendation_17_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation_v1_4(TestRecommendationBase):
    recommendation_url = (
        "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation"
    )
    recommendation_package_version = "v1.4.0"
    recommendation_parser_version = 2  # for recommendation package version >= 1.4

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

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The recommendation uses ~(ICU | VENOUS_THROMBOSIS) as a population criterion, and at least
        ICU ends within a day, i.e. the day has both POSITIVE and NEGATIVE intervals for ICU.
        Now, for population, POSITIVE takes precedence over NEGATIVE, so the day is marked as POSITIVE.
        However, the expected value for the day is determined on basis of the whole day, and inverting the POSITIVE
        label gets NEGATIVE (as a label for the whole day), although there is actually both NEGATIVE and POSITIVE during
        the day, so inversion would lead to still POSITIVE and NEGATIVE being there, and so the day would be POSITIVE,
        if we would consider within-day granularity, which, unfortunately, we do not (here, in testing!).

        So we just forcibly set the last entry of POSITIVE to NEGATIVE.
        """
        cols = [("ICU", "=")]

        def set_last_positive_to_negative(group):
            idx_positive = group[group == IntervalType.POSITIVE].index
            if not idx_positive.empty:
                last_positive_idx = idx_positive[-1]
                group.at[last_positive_idx] = IntervalType.NEGATIVE
            return group

        for c in cols:
            df[c] = (
                df[c]
                .groupby(level="person_id", group_keys=False)
                .apply(set_last_positive_to_negative)
            )

        return df

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
    recommendation_parser_version = 1  # for recommendation package version < 1.4

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

    def _modify_criteria_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The recommendation uses ~(ICU | VENOUS_THROMBOSIS) as a population criterion, and at least
        ICU ends within a day, i.e. the day has both POSITIVE and NEGATIVE intervals for ICU.
        Now, for population, POSITIVE takes precedence over NEGATIVE, so the day is marked as POSITIVE.
        However, the expected value for the day is determined on basis of the whole day, and inverting the POSITIVE
        label gets NEGATIVE (as a label for the whole day), although there is actually both NEGATIVE and POSITIVE during
        the day, so inversion would lead to still POSITIVE and NEGATIVE being there, and so the day would be POSITIVE,
        if we would consider within-day granularity, which, unfortunately, we do not (here, in testing!).

        So we just forcibly set the last entry of POSITIVE to NEGATIVE.
        """
        cols = [("ICU", "=")]

        def set_last_positive_to_negative(group):
            idx_positive = group[group == IntervalType.POSITIVE].index
            if not idx_positive.empty:
                last_positive_idx = idx_positive[-1]
                group.at[last_positive_idx] = IntervalType.NEGATIVE
            return group

        for c in cols:
            df[c] = (
                df[c]
                .groupby(level="person_id", group_keys=False)
                .apply(set_last_positive_to_negative)
            )

        return df

    def test_recommendation_17_therapeutic_anticoagulation(
        self, setup_testdata
    ) -> None:
        self.recommendation_test_runner(setup_testdata)
