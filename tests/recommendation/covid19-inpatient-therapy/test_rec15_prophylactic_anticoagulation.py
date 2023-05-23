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


class TestRecommendation15ProphylacticAnticoagulation(TestRecommendationBase):
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
            "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
        )

        return f"{base_url}{recommendation_url}"

    @pytest.fixture
    def population_intervention(self) -> dict:
        population = {
            "COVID19": concepts.COVID19,
            "VENOUS_THROMBOSIS": concepts.VENOUS_THROMBOSIS,
            "HIT2": concepts.HEPARIN_INDUCED_THROMBOCYTOPENIA_WITH_THROMBOSIS,
            "HEPARIN_ALLERGY": concepts.ALLERGY_HEPARIN,
            "HEPARINOID_ALLERGY": concepts.ALLERGY_HEPARINOID,
            "THROMBOCYTOPENIA": concepts.THROMBOCYTOPENIA,
        }

        interventions = {
            "DALTEPARIN": concepts.DALTEPARIN,
            "ENOXAPARIN": concepts.ENOXAPARIN,
            "NADROPARIN_LOW_WEIGHT": concepts.NADROPARIN,
            "NADROPARIN_HIGH_WEIGHT": concepts.NADROPARIN,
            "CERTOPARIN": concepts.CERTOPARIN,
            "FONDAPARINUX": concepts.FONDAPARINUX,
        }

        return population | interventions

    @pytest.fixture
    def population_intervention_groups(self, population_intervention: dict) -> dict:
        return {
            "AntithromboticProphylaxisWithLWMH": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY | THROMBOCYTOPENIA)",
                "intervention": "DALTEPARIN | ENOXAPARIN | NADROPARIN_LOW_WEIGHT | NADROPARIN_HIGH_WEIGHT | CERTOPARIN",
            },
            "AntithromboticProphylaxisWithFondaparinux": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY | THROMBOCYTOPENIA)",
                "intervention": "FONDAPARINUX",
            },
            "NoAntithromboticProphylaxis": {
                "population": "COVID19 & VENOUS_THROMBOSIS",
                "intervention": "~(DALTEPARIN | ENOXAPARIN | NADROPARIN_LOW_WEIGHT | NADROPARIN_HIGH_WEIGHT | CERTOPARIN | FONDAPARINUX)",
            },
        }

    @pytest.fixture
    def invalid_combinations(self, population_intervention: dict) -> str:
        return "NADROPARIN_HIGH_WEIGHT & NADROPARIN_LOW_WEIGHT"

    def test_recommendation_15_prophylactic_anticoagulation(
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
            plan_names=list(population_intervention_groups.keys()),
        )
