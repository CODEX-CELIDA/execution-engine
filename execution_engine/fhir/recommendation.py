import logging

from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from fhir.resources.plandefinition import (
    PlanDefinition,
    PlanDefinitionAction,
    PlanDefinitionGoal,
)

from .client import FHIRClient


class Recommendation:
    """CPG-on-EBM-on-FHIR Recommendation."""

    class Action:
        """
        Action Definition contained in the PlanDefinition

        Helper class to separate the action definition (contained in PlanDefinition), which specifies the goals,
        type etc., from the activity defininition (contained in ActivityDefinition), which may be referenced by the
        action in the definitionCanonical elements, but may also be null.
        """

        def __init__(
            self,
            action_def: PlanDefinitionAction,
            goals: list[PlanDefinitionGoal],
            fhir_connector: FHIRClient,
        ) -> None:
            """Create a new action from a FHIR PlanDefinition.action."""
            self._action: PlanDefinitionAction = action_def
            self._activity: ActivityDefinition | None = None

            # an action must not necessarily contain an activity definition (e.g. ventilator management)
            if action_def.definitionCanonical is not None:
                self._activity = fhir_connector.get_resource(
                    "ActivityDefinition", action_def.definitionCanonical
                )

            if action_def.goalId is None:
                self._goals = []
            else:
                self._goals = [g for g in goals if g.id in action_def.goalId]

        @property
        def action(self) -> PlanDefinitionAction:
            """Get the FHIR PlanDefinition.action."""
            return self._action

        @property
        def activity(self) -> ActivityDefinition | None:
            """Get the ActivityDefinition for this action."""
            return self._activity

        @property
        def goals(self) -> list[PlanDefinitionGoal]:
            """Get the goals for this action."""
            return self._goals

    def __init__(self, canonical_url: str, fhir_connector: FHIRClient):
        self.canonical_url = canonical_url
        self.fhir = fhir_connector

        self._recommendation = None
        self._population = None
        self._actions: list[Recommendation.Action] = []
        self._goals: list[PlanDefinitionGoal] = []

        self.load()

    def load(self) -> None:
        """
        Load the recommendation from the FHIR server.
        """

        plan_def = self.get_recommendation(self.canonical_url)
        ev = self.fhir.get_resource("EvidenceVariable", plan_def.subjectCanonical)

        self._recommendation = plan_def
        self._population = ev
        self._actions = [
            Recommendation.Action(action, goals=plan_def.goal, fhir_connector=self.fhir)
            for action in plan_def.action
        ]
        self._goals = plan_def.goal

        logging.info("Recommendation loaded.")

    def get_recommendation(self, canonical_url: str) -> PlanDefinition:
        """Read the PlanDefinition resource from the FHIR server."""
        return self.fhir.get_resource("PlanDefinition", canonical_url)

    @property
    def population(self) -> EvidenceVariable:
        """
        The population for the recommendation.
        """
        return self._population

    @property
    def actions(self) -> list[Action]:
        """
        The actions for the recommendation.
        """
        return self._actions

    @property
    def goals(self) -> list[PlanDefinitionGoal]:
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
