from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.converter.action.assessment import AssessmentAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.ventilator_management import (
    VentilatorManagementAction,
)
from execution_engine.converter.characteristic.allergy import AllergyCharacteristic
from execution_engine.converter.characteristic.condition import ConditionCharacteristic
from execution_engine.converter.characteristic.episode_of_care import (
    EpisodeOfCareCharacteristic,
)
from execution_engine.converter.characteristic.observation import (
    ObservationCharacteristic,
)
from execution_engine.converter.characteristic.procedure import ProcedureCharacteristic
from execution_engine.converter.characteristic.radiology import RadiologyCharacteristic
from execution_engine.converter.characteristic.ventilation_observable import (
    VentilationObservableCharacteristic,
)
from execution_engine.converter.converter import CriterionConverterFactory
from execution_engine.converter.goal.assessment_scale import AssessmentScaleGoal
from execution_engine.converter.goal.laboratory_value import LaboratoryValueGoal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)
from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.converter.parser.fhir_parser_v1 import FhirRecommendationParserV1
from execution_engine.converter.parser.fhir_parser_v2 import FhirRecommendationParserV2


class FhirRecommendationParserFactory:
    """Factory to instantiate the correct FhirParser based on the version string."""

    default_converters = {
        "characteristic": [
            ConditionCharacteristic,
            AllergyCharacteristic,
            RadiologyCharacteristic,
            ProcedureCharacteristic,
            EpisodeOfCareCharacteristic,
            VentilationObservableCharacteristic,
            ObservationCharacteristic,
        ],
        "action": [
            DrugAdministrationAction,
            VentilatorManagementAction,
            BodyPositioningAction,
            AssessmentAction,
        ],
        "goal": [LaboratoryValueGoal, VentilatorManagementGoal, AssessmentScaleGoal],
    }

    def __init__(self, builder: ExecutionEngineBuilder | None = None):
        self.characteristic_converters = CriterionConverterFactory()
        self.action_converters = CriterionConverterFactory()
        self.goal_converters = CriterionConverterFactory()

        if builder is not None:
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
