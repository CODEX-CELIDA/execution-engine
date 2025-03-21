from abc import ABC, abstractmethod
from typing import Callable, Type

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicTimeFromEvent,
)
from fhir.resources.extension import Extension
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

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
        relative_time_converters: TemporalIndicatorConverterFactory,
    ):
        self.characteristics_converters = characteristic_converters
        self.action_converters = action_converters
        self.goal_converters = goal_converters
        self.time_from_event_converters = time_from_event_converters
        self.relative_time_converters = relative_time_converters

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

    @abstractmethod
    def parse_action_combination_method(
        self, action_parent: PlanDefinition | PlanDefinitionAction
    ) -> Type[logic.BooleanFunction] | Callable:
        """
        Parses the action combination method of a Recommendation (PlanDefinition) and returns the corresponding
        combination method in form of a logical expression.
        """
        raise NotImplementedError()

    def parse_time_from_event(
        self,
        tfes: list[EvidenceVariableCharacteristicTimeFromEvent],
    ) -> list[logic.BaseExpr]:
        """
        Parses `timeFromEvent` elements and converts them into interval-based logical criteria.
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_relative_time(
        self,
        relative_time: list[Extension],
    ) -> list[logic.BaseExpr]:
        """
        Parses `extension[relativeTime]` elements and converts them into interval-based logical criteria.
        """
        raise NotImplementedError()

    def parse_timing(
        self, characteristic: EvidenceVariableCharacteristic, expr: logic.BaseExpr
    ) -> logic.BaseExpr:
        """
        Applies temporal constraints to a given criterion expression based on `timeFromEvent` and
        the relativeTime extension elements.
        """
        raise NotImplementedError()
