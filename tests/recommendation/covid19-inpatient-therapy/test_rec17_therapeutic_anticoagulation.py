import datetime

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util import TimeRange
from tests._testdata import concepts
from tests.recommendation.test_recommendation import TestRecommendationBase


@pytest.mark.recommendation
class TestRecommendation17TherapeuticAnticoagulation(TestRecommendationBase):
    @pytest.fixture
    def visit_datetime(self) -> TimeRange:
        return TimeRange(
            start="2023-03-01 07:00:00", end="2023-03-31 22:00:00", name="visit"
        )

    @pytest.fixture
    def observation_window(self, visit_datetime: TimeRange) -> TimeRange:
        return TimeRange(
            start=visit_datetime.start - datetime.timedelta(days=3),
            end=visit_datetime.end + datetime.timedelta(days=3),
            name="observation",
        )

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
    def population_intervention(self) -> dict:
        population = {
            "COVID19": concepts.COVID19,
            "VENOUS_THROMBOSIS": concepts.VENOUS_THROMBOSIS,
            "ICU": concepts.INTENSIVE_CARE,
            "D_DIMER": concepts.LAB_DDIMER,
        }

        interventions = {
            "HEPARIN": concepts.HEPARIN,
            "DALTEPARIN": concepts.DALTEPARIN,
            "NADROPARIN_LOW_WEIGHT": concepts.NADROPARIN,
            "NADROPARIN_HIGH_WEIGHT": concepts.NADROPARIN,
            "ENOXAPARIN": concepts.ENOXAPARIN,
            "CERTOPARIN": concepts.CERTOPARIN,
            "FONDAPARINUX": concepts.FONDAPARINUX,
            "ARGATROBAN": concepts.ARGATROBAN,
        }

        return population | interventions

    @pytest.fixture
    def population_intervention_groups(self, population_intervention: dict) -> dict:
        return {
            "Therapeutic_Anticoagulation_No_Renal_Function_Impairment": {
                "population": "COVID19 & ~ICU & ~VENOUS_THROMBOSIS & D_DIMER",
                "intervention": "HEPARIN | DALTEPARIN | NADROPARIN_LOW_WEIGHT | NADROPARIN_HIGH_WEIGHT | ENOXAPARIN | CERTOPARIN | FONDAPARINUX | ARGATROBAN",
            },
        }

    @pytest.fixture
    def invalid_combinations(self, population_intervention: dict) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_17_therapeutic_anticoagulation(
        self,
        db_session: sessionmaker,
        population_intervention_groups: dict,
        criteria_extended: pd.DataFrame,
        observation_window: TimeRange,
        recommendation_url: str,
    ) -> None:
        assert self.recommendation_test_runner(
            recommendation_url=recommendation_url,
            observation_window=observation_window,
            criteria_extended=criteria_extended,
        )
