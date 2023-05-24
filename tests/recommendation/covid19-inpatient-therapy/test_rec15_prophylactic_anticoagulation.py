import pandas as pd
import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from execution_engine.util import TimeRange
from tests.recommendation.test_recommendation import TestRecommendationBase


class TestRecommendation15ProphylacticAnticoagulation(TestRecommendationBase):
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
        return {
            "AntithromboticProphylaxisWithLWMH": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & ~(HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY | THROMBOCYTOPENIA)",
                "intervention": "DALTEPARIN< | ENOXAPARIN< | NADROPARIN_LOW_WEIGHT< | NADROPARIN_HIGH_WEIGHT< | CERTOPARIN<",
            },
            "AntithromboticProphylaxisWithFondaparinux": {
                "population": "COVID19 & ~VENOUS_THROMBOSIS & (HIT2 | HEPARIN_ALLERGY | HEPARINOID_ALLERGY | THROMBOCYTOPENIA)",
                "intervention": "FONDAPARINUX_PROPHYLACTIC=",
            },
            "NoAntithromboticProphylaxis": {
                "population": "COVID19 & VENOUS_THROMBOSIS",
                "intervention": "~(DALTEPARIN< | ENOXAPARIN< | NADROPARIN_LOW_WEIGHT< | NADROPARIN_HIGH_WEIGHT< | CERTOPARIN< | FONDAPARINUX_PROPHYLACTIC=)",
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
    ) -> None:

        self.recommendation_test_runner(
            recommendation_url=recommendation_url,
            observation_window=observation_window,
            criteria_extended=criteria_extended,
        )
