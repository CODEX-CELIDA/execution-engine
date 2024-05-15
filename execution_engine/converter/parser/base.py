from abc import ABC, abstractmethod

from fhir.resources.evidencevariable import EvidenceVariable

from execution_engine import fhir
from execution_engine.converter.action.assessment import AssessmentAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.ventilator_management import (
    VentilatorManagementAction,
)
from execution_engine.converter.characteristic.allergy import AllergyCharacteristic
from execution_engine.converter.characteristic.combination import (
    CharacteristicCombination,
)
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
from execution_engine.omop.criterion.combination import CriterionCombination


class FhirRecommendationParserInterface(ABC):
    """Define a common interface for all FHIR parsers."""

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

    @abstractmethod
    def parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        raise NotImplementedError()

    @abstractmethod
    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> CriterionCombination:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """
        raise NotImplementedError()
