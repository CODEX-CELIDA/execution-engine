from execution_engine.converter.parser.base import FhirParserInterface
from execution_engine.converter.parser.fhir_parser_v1 import FhirParserV1
from execution_engine.converter.parser.fhir_parser_v2 import FhirParserV2
from execution_engine.util import version


class FhirParserFactory:
    """Factory to instantiate the correct FhirParser based on the version string."""

    @staticmethod
    def get_parser(version_str: str) -> FhirParserInterface:
        """
        Return the correct FhirParser based on the version string.
        """
        version_map = {
            "1": FhirParserV1,
            "2": FhirParserV2,
        }

        if version.is_version_below(version_str, "1.4"):
            version_str = "1"
        else:
            version_str = "2"

        parser_class = version_map.get(version_str)

        if parser_class:
            return parser_class()
        else:
            raise ValueError(f"No parser available for FHIR version {version_str}")
