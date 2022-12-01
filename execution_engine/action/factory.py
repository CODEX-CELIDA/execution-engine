from typing import Type

from ..fhir.recommendation import Recommendation
from .abstract import AbstractAction


class ActionFactory:
    """Factory for creating Action objects from PlanDefinition.action."""

    def __init__(self) -> None:
        self._action_types: list[Type[AbstractAction]] = []

    def register_action_type(self, action: Type[AbstractAction]) -> None:
        """Register a new action type."""
        self._action_types.append(action)

    def get_action(self, action_def: Recommendation.Action) -> AbstractAction:
        """Get the action type for the given CodeableConcept from PlanDefinition.action.code."""
        for action in self._action_types:
            if action.valid(action_def):
                return action.from_fhir(action_def)
        raise ValueError("No action type matched the FHIR definition.")
