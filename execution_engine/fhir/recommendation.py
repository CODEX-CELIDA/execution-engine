import logging
from typing import List

from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction, PlanDefinitionGoal

from .client import FHIRClient


class Recommendation:
    """CPG-on-EBM-on-FHIR Recommendation."""

    class Action:

        def __init__(self, action_def: PlanDefinitionAction, fhir_connector: FHIRClient) -> None:
            """Create a new action from a FHIR PlanDefinition.action."""
            self._action: PlanDefinitionAction = action_def
            self._activity: ActivityDefinition = fhir_connector.get_resource("ActivityDefinition",
                                                                             action_def.definitionCanonical)

        @property
        def action(self) -> PlanDefinitionAction:
            """Get the FHIR PlanDefinition.action."""
            return self._action

        @property
        def activity(self) -> ActivityDefinition:
            """Get the ActivityDefinition for this action."""
            return self._activity

    def __init__(self, canonical_url: str, fhir_connector: FHIRClient):
        self.canonical_url = canonical_url
        self.fhir = fhir_connector

        self._recommendation = None
        self._population = None
        self._actions: List[Recommendation.Action] = []
        self._goals: List[PlanDefinitionGoal] = []

        self.load()

    def load(self) -> None:
        """
        Load the recommendation from the FHIR server.
        """

        plan_def = self.get_recommendation(self.canonical_url)
        ev = self.fhir.get_resource("EvidenceVariable", plan_def.subjectCanonical)

        self._recommendation = plan_def
        self._population = ev
        self._actions = [Recommendation.Action(action, self.fhir) for action in plan_def.action]
        self._goals = plan_def.goal

        logging.info("Recommendation loaded.")

    def get_recommendation(self, canonical_url: str) -> PlanDefinition:
        """Read the PlanDefinition resource from the FHIR server."""
        return self.fhir.get_resource("PlanDefinition", canonical_url)

    def get_activity_definitions(
        self, actions: List[PlanDefinitionAction]
    ) -> List[ActivityDefinition]:
        """Read the ActivityDefinition resources from the FHIR server."""
        return [
            self.fhir.get_resource("ActivityDefinition", action.definitionCanonical)
            for action in actions
        ]

    @property
    def population(self) -> EvidenceVariable:
        """
        The population for the recommendation.
        """
        return self._population

    @property
    def actions(self) -> List[PlanDefinitionGoal]:
        """
        The actions for the recommendation.
        """
        return self._actions

    @property
    def goals(self) -> List[ActivityDefinition]:
        """
        The actions for the recommendation.
        """
        return self._goals

    @staticmethod
    def is_combination_definition(
        characteristic: EvidenceVariableCharacteristic,
    ) -> bool:
        """
        Determine if the characteristic is a combination of other characteristics.

        EvidenceVariable.characteristic are either a single characteristic defined using
        EvidenceVariableCharacteristic.definitionByTypeAndValue or a combination of other
        characteristics defined using EvidenceVariableCharacteristic.definitionByCombination.
        """
        return characteristic.definitionByCombination is not None
