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
from execution_engine.converter.characteristic.laboratory import (
    LaboratoryCharacteristic,
)
from execution_engine.converter.characteristic.procedure import ProcedureCharacteristic
from execution_engine.converter.characteristic.radiology import RadiologyCharacteristic
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

    def __init__(self, builder: ExecutionEngineBuilder | None = None):
        self.characteristic_converters = self.init_characteristics_factory()
        self.action_converters = self.init_action_factory()
        self.goal_converters = self.init_goal_factory()

        if builder is not None:
            for converter in builder.characteristic_converters:
                self.characteristic_converters.register(converter)

            for converter in builder.action_converters:
                self.action_converters.register(converter)

            for converter in builder.goal_converters:
                self.goal_converters.register(converter)

    @staticmethod
    def init_characteristics_factory() -> CriterionConverterFactory:
        """
        Initialize the characteristic factory with the characteristics that are supported by the parser.
        """
        cf = CriterionConverterFactory()
        cf.register(ConditionCharacteristic)
        cf.register(AllergyCharacteristic)
        cf.register(RadiologyCharacteristic)
        cf.register(ProcedureCharacteristic)
        cf.register(EpisodeOfCareCharacteristic)
        # cf.register(VentilationObservableCharacteristic) # fixme: implement (valueset retrieval / caching)
        cf.register(LaboratoryCharacteristic)

        return cf

    @staticmethod
    def init_action_factory() -> CriterionConverterFactory:
        """
        Initialize the action factory with the actions that are supported by the parser.
        """
        af = CriterionConverterFactory()
        af.register(DrugAdministrationAction)
        af.register(VentilatorManagementAction)
        af.register(BodyPositioningAction)
        af.register(AssessmentAction)

        return af

    @staticmethod
    def init_goal_factory() -> CriterionConverterFactory:
        """
        Initialize the goal factory with the goals that are supported by the parser.
        """
        gf = CriterionConverterFactory()
        gf.register(LaboratoryValueGoal)
        gf.register(VentilatorManagementGoal)
        gf.register(AssessmentScaleGoal)

        return gf

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
