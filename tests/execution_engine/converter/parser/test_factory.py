import pytest

from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.converter.converter import CriterionConverterFactory
from execution_engine.converter.parser.factory import FhirRecommendationParserFactory
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.converter.parser.fhir_parser_v2 import FhirRecommendationParserV2


class TestFhirRecommendationParserFactory:
    def setup_method(self):
        self.builder = ExecutionEngineBuilder()
        self.factory = FhirRecommendationParserFactory(self.builder)

    def test_init_with_builder(self):
        """Test the factory initialization with a custom builder."""
        assert isinstance(
            self.factory.characteristic_converters, CriterionConverterFactory
        )
        assert isinstance(self.factory.action_converters, CriterionConverterFactory)
        assert isinstance(self.factory.goal_converters, CriterionConverterFactory)

    def test_get_parser_version_1(self):
        factory = FhirRecommendationParserFactory()
        parser = factory.get_parser(1)
        assert isinstance(
            parser, FhirRecommendationParserV1
        ), "Version 1 should return FhirRecommendationParserV1 instance"

    def test_get_parser_version_2(self):
        factory = FhirRecommendationParserFactory()
        parser = factory.get_parser(2)
        assert isinstance(
            parser, FhirRecommendationParserV2
        ), "Version 2 should return FhirRecommendationParserV2 instance"

    def test_get_parser_unsupported_version(self):
        factory = FhirRecommendationParserFactory()
        with pytest.raises(ValueError) as exc_info:
            factory.get_parser(999)
        assert "No parser available for FHIR version 999" in str(exc_info.value)

    def test_get_parser_with_builder(self):
        builder = ExecutionEngineBuilder()
        factory = FhirRecommendationParserFactory(builder=builder)
        parser = factory.get_parser(1)
        assert isinstance(
            parser, FhirRecommendationParserV1
        ), "Builder integration should still return the correct parser version"
