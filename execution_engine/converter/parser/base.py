from abc import ABC, abstractmethod

from fhir.resources.evidencevariable import EvidenceVariable

from execution_engine import fhir
from execution_engine.converter.criterion import CriterionConverterFactory
from execution_engine.converter.temporal import TemporalIndicatorConverterFactory
from execution_engine.util import logic as logic


class FhirRecommendationParserInterface(ABC):
    """Define a common interface for all FHIR parsers."""

    def __init__(
        self,
        characteristic_converters: CriterionConverterFactory,
        action_converters: CriterionConverterFactory,
        goal_converters: CriterionConverterFactory,
        time_from_event_converters: TemporalIndicatorConverterFactory,
    ):
        self.characteristics_converters = characteristic_converters
        self.action_converters = action_converters
        self.goal_converters = goal_converters
        self.time_from_event_converters = time_from_event_converters

    @abstractmethod
    def parse_characteristics(self, ev: EvidenceVariable) -> logic.BooleanFunction:
        """
        Parses the EvidenceVariable characteristics and returns a BooleanFunction
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> logic.BooleanFunction:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """
        raise NotImplementedError()
