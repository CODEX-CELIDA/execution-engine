from abc import ABC, abstractmethod

from fhir.resources.evidencevariable import EvidenceVariable

from execution_engine import fhir
from execution_engine.converter.characteristic.combination import (
    CharacteristicCombination,
)
from execution_engine.converter.converter import CriterionConverterFactory
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)


class FhirRecommendationParserInterface(ABC):
    """Define a common interface for all FHIR parsers."""

    def __init__(
        self,
        characteristic_converters: CriterionConverterFactory,
        action_converters: CriterionConverterFactory,
        goal_converters: CriterionConverterFactory,
    ):
        self.characteristics_converters = characteristic_converters
        self.action_converters = action_converters
        self.goal_converters = goal_converters

    @abstractmethod
    def parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        raise NotImplementedError()

    @abstractmethod
    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> LogicalCriterionCombination:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """
        raise NotImplementedError()
