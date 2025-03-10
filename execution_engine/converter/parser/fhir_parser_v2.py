from typing import Callable, Type

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

from execution_engine import constants
from execution_engine.converter.criterion import get_extension_by_url
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.fhir.util import get_coding
from execution_engine.util import logic as logic


class FhirRecommendationParserV2(FhirRecommendationParserV1):
    """
    Parses a recommendation in CPG-on-EBMonFHIR format with version >= v0.8.

    Starting with version 0.8, the method of combination of actions is read from an extension to PlanDefinition.
    Also, nested actions are parsed.
    """

    def parse_action_combination_method(
        self, action_parent: PlanDefinition | PlanDefinitionAction
    ) -> Type[logic.BooleanFunction] | Callable:
        """
        Parses the action combination method from an extension to a PlanDefinition or PlanDefinitionAction.
        """
        ext = get_extension_by_url(
            action_parent, constants.EXT_ACTION_COMBINATION_METHOD
        )

        method: CodeableConcept = get_extension_by_url(
            ext, "method"
        ).valueCodeableConcept

        method_code = get_coding(
            method, system_uri=constants.CS_ACTION_COMBINATION_METHOD
        ).code

        try:
            threshold = get_extension_by_url(ext, "threshold").valuePositiveInt
        except ValueError:
            threshold = None

        expr: Type[logic.BooleanFunction] | Callable

        match method_code:
            case "all":
                expr = logic.And
            case "any":
                expr = logic.Or
            case "at-most":
                expr = lambda *args: logic.MaxCount(*args, threshold=threshold)
            case "exactly":
                expr = lambda *args: logic.ExactCount(*args, threshold=threshold)
            case "at-least":
                expr = lambda *args: logic.MinCount(*args, threshold=threshold)
            case "one-or-more":
                expr = lambda *args: logic.MinCount(*args, threshold=1)
            case _:
                raise ValueError(f"Invalid action combination method: {method_code}")

        return expr
