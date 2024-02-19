from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.converter.parser.fhir_parser_v2 import FhirRecommendationParserV2
from execution_engine.util import version


class FhirRecommendationParserFactory:
    """Factory to instantiate the correct FhirParser based on the version string."""

    @staticmethod
    def get_parser(version_str: str) -> FhirRecommendationParserInterface:
        """
        Return the correct FhirParser based on the version string.
        """
        version_map = {
            "1.3": FhirRecommendationParserV1,
            "1.4": FhirRecommendationParserV2,
            "latest": FhirRecommendationParserV2,
        }

        if version_str in version_map:
            pass
        elif version.is_version_below(version_str, "1.4"):
            version_str = "1.3"
        else:
            version_str = "1.4"

        parser_class = version_map.get(version_str)

        if parser_class:
            return parser_class()
        else:
            raise ValueError(f"No parser available for FHIR version {version_str}")
