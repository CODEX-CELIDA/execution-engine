from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

from execution_engine import constants
from execution_engine.constants import CohortCategory
from execution_engine.converter.converter import get_extension_by_url
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.omop.criterion.combination import CriterionCombination


class FhirRecommendationParserV2(FhirRecommendationParserV1):
    """
    Parses a recommendation in CPG-on-EBMonFHIR format with version >= v0.8.

    Starting with version 0.8, the method of combination of actions is read from an extension to PlanDefinition.
    Also, nested actions are parsed.
    """

    def parse_action_combination_method(
        self, action_parent: PlanDefinition | PlanDefinitionAction
    ) -> CriterionCombination:
        """
        Parses the action combination method from an extension to a PlanDefinition or PlanDefinitionAction.
        """
        ext = get_extension_by_url(
            action_parent, constants.EXT_ACTION_COMBINATION_METHOD
        )

        method: CodeableConcept = get_extension_by_url(
            ext, "method"
        ).valueCodeableConcept
        if method.system != constants.CS_ACTION_COMBINATION_METHOD:
            raise ValueError(f"Invalid action combination method: {method.system}")
        if len(method.coding) != 1:
            raise ValueError(
                f"Expected exactly one coding in action combination method: {method.coding}"
            )
        method_code = method.coding[0].code

        try:
            threshold = get_extension_by_url(ext, "threshold").valuePositiveInt
        except ValueError:
            threshold = None

        match method_code:
            case "all":
                operator = CriterionCombination.Operator("AND")
            case "any":
                operator = CriterionCombination.Operator("OR")
            case "at-most":
                operator = CriterionCombination.Operator("AT_MOST", threshold=threshold)
            case "exactly":
                operator = CriterionCombination.Operator("EXACTLY", threshold=threshold)
            case "at-least":
                operator = CriterionCombination.Operator(
                    "AT_LEAST", threshold=threshold
                )
            case "one-or-more":
                operator = CriterionCombination.Operator("AT_LEAST", threshold=1)
            case _:
                raise ValueError(f"Invalid action combination method: {method_code}")

        return CriterionCombination(
            name="intervention_actions",
            category=CohortCategory.INTERVENTION,
            exclude=False,
            operator=operator,
        )
