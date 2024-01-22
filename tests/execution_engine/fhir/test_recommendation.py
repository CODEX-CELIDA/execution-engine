from unittest.mock import patch

import pytest
from fhir.resources import construct_fhir_element

from execution_engine.constants import CS_PLAN_DEFINITION_TYPE
from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir.recommendation import Recommendation, RecommendationPlan


class RecommendationFixtures:
    @pytest.fixture
    def test_class(self):
        raise NotImplementedError("Test class must implement this fixture.")

    @pytest.fixture
    def mock_load(self, test_class):
        with patch.object(test_class, "load") as _fixture:
            yield _fixture

    @pytest.fixture
    def mock_fetch_resource_unknown(self):
        plan_definition = construct_fhir_element(
            "PlanDefinition",
            {
                "status": "draft",
                "action": [],
                "type": {
                    "coding": [
                        {"code": "unknown-type", "system": CS_PLAN_DEFINITION_TYPE}
                    ]
                },
            },
        )
        with patch.object(
            FHIRClient, "fetch_resource", return_value=plan_definition
        ) as _fixture:
            yield _fixture

    @pytest.fixture
    def mock_fetch_resource_no_partOf(self):
        plan_definition = construct_fhir_element(
            "PlanDefinition",
            {
                "status": "draft",
                "action": [],
                "type": {
                    "coding": [{"code": "eca-rule", "system": CS_PLAN_DEFINITION_TYPE}]
                },
            },
        )
        with patch.object(
            FHIRClient, "fetch_resource", return_value=plan_definition
        ) as _fixture:
            yield _fixture


class TestRecommendation(RecommendationFixtures):
    @pytest.fixture
    def test_class(self):
        return Recommendation

    @pytest.fixture
    def mock_fetch_recommendation(self, test_class):
        plan_definition = construct_fhir_element(
            "PlanDefinition", {"status": "draft", "action": []}
        )
        with patch.object(
            test_class, "fetch_recommendation", return_value=plan_definition
        ) as _fixture:
            yield _fixture

    def test_properties_before_load(self, mock_load):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"
        rec = Recommendation(
            canonical_url,
            package_version="latest",
            fhir_connector=FHIRClient("http://fhir.example.com"),
        )

        # Test
        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.name

        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.title

        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.url

        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.version

        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.description

    def test_properties_without_data(self, mock_fetch_recommendation):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"
        rec = Recommendation(
            canonical_url,
            package_version="latest",
            fhir_connector=FHIRClient("http://fhir.example.com"),
        )

        # Test
        with pytest.raises(ValueError, match=r"Recommendation has no name."):
            _ = rec.name

        with pytest.raises(ValueError, match=r"Recommendation has no title."):
            _ = rec.title

        with pytest.raises(ValueError, match=r"Recommendation has no URL."):
            _ = rec.url

        with pytest.raises(ValueError, match=r"Recommendation has no version."):
            _ = rec.version

        with pytest.raises(ValueError, match=r"Recommendation has no description."):
            _ = rec.description

    def test_recommendation_load_with_unknown_type(self, mock_fetch_resource_unknown):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"

        # Test
        with pytest.raises(
            ValueError, match=r"Unknown recommendation type: unknown-type"
        ):
            _ = Recommendation(
                canonical_url,
                package_version="latest",
                fhir_connector=FHIRClient("http://fhir.example.com"),
            )

    def test_recommendation_fetch_with_no_partof_extension(
        self, mock_fetch_resource_no_partOf
    ):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"

        # Test
        with pytest.raises(
            ValueError,
            match=r"No partOf extension found in PlanDefinition, can't fetch recommendation.",
        ):
            _ = Recommendation(
                canonical_url,
                package_version="latest",
                fhir_connector=FHIRClient("http://fhir.example.com"),
            )


class TestRecommendationPlan(RecommendationFixtures):
    @pytest.fixture
    def test_class(self):
        return RecommendationPlan

    @pytest.fixture
    def mock_fetch_recommendation_plan(self):
        plan_definition = construct_fhir_element(
            "PlanDefinition", {"status": "draft", "action": []}
        )
        with patch.object(
            RecommendationPlan,
            "fetch_recommendation_plan",
            return_value=plan_definition,
        ) as _fixture:
            yield _fixture

    @pytest.fixture
    def mock_fetch_resource_goals(self):
        plan_definition = construct_fhir_element(
            "PlanDefinition",
            {
                "status": "draft",
                "action": [],
                "type": {
                    "coding": [{"code": "eca-rule", "system": CS_PLAN_DEFINITION_TYPE}]
                },
                "goal": [{"description": {"text": "http://test.com/Goal/123"}}],
            },
        )
        with patch.object(
            FHIRClient, "fetch_resource", return_value=plan_definition
        ) as _fixture:
            yield _fixture

    def test_properties_before_load(self, mock_load):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"
        rec = RecommendationPlan(
            canonical_url,
            package_version="v1.3.0",
            fhir_connector=FHIRClient("http://fhir.example.com"),
        )

        # Test
        with pytest.raises(ValueError, match=r"Recommendation not loaded."):
            _ = rec.name

    def test_properties_without_data(
        self, mock_fetch_recommendation_plan, mock_fetch_resource_goals
    ):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"
        rec = RecommendationPlan(
            canonical_url,
            package_version="v1.3.0",
            fhir_connector=FHIRClient("http://fhir.example.com"),
        )

        # Test
        with pytest.raises(ValueError, match=r"Recommendation has no name."):
            _ = rec.name

    def test_recommendation_load_with_unknown_type(self, mock_fetch_resource_unknown):
        # Setup
        canonical_url = "http://test.com/PlanDefinition/123"

        # Test
        with pytest.raises(
            ValueError, match=r"Unknown recommendation type: unknown-type"
        ):
            _ = RecommendationPlan(
                canonical_url,
                package_version="v1.3.0",
                fhir_connector=FHIRClient("http://fhir.example.com"),
            )

    def test_goals_property(self, mock_fetch_resource_goals):
        canonical_url = "http://test.com/PlanDefinition/123"

        rec = RecommendationPlan(
            canonical_url,
            package_version="v1.3.0",
            fhir_connector=FHIRClient("http://fhir.example.com"),
        )

        assert rec.goals[0].description.text == "http://test.com/Goal/123"
