from execution_engine.builder import (
    ExecutionEngineBuilder,
    default_execution_engine_builder,
)
from execution_engine.converter.converter import CriterionConverterFactory
from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.converter.parser.fhir_parser_v2 import FhirRecommendationParserV2


class FhirRecommendationParserFactory:
    """Factory to instantiate the correct FhirParser based on the version string."""

    def __init__(self, builder: ExecutionEngineBuilder | None = None):
        self.characteristic_converters = CriterionConverterFactory()
        self.action_converters = CriterionConverterFactory()
        self.goal_converters = CriterionConverterFactory()

        if builder is None:
            builder = default_execution_engine_builder()

        for converter in builder.characteristic_converters:
            self.characteristic_converters.register(converter)

        for converter in builder.action_converters:
            self.action_converters.register(converter)

        for converter in builder.goal_converters:
            self.goal_converters.register(converter)

    def get_parser(self, parser_version: int) -> FhirRecommendationParserInterface:
        """
        Return the correct FhirParser based on the version string.
        """
        match parser_version:
            case 1:
                parser_class = FhirRecommendationParserV1
            case 2:
                parser_class = FhirRecommendationParserV2
            case _:
                raise ValueError(
                    f"No parser available for FHIR version {parser_version}"
                )

        return parser_class(
            characteristic_converters=self.characteristic_converters,
            action_converters=self.action_converters,
            goal_converters=self.goal_converters,
        )
