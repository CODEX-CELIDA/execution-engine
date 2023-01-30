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

from ..constants import CS_PLAN_DEFINITION_TYPE, EXT_CPG_PARTOF
from .client import FHIRClient
from .util import get_coding, get_extension


class Recommendation:
    """CPG-on-EBM-on-FHIR Recommendation."""

    def __init__(self, url: str, fhir_connector: FHIRClient) -> None:
        self.fhir = fhir_connector

        self._recommendation: PlanDefinition | None = None
        self._recommendation_plans: list[RecommendationPlan] = []

        self.load(url)

    @property
    def name(self) -> str:
        """
        Return the name of the recommendation.
        """
        if self._recommendation is None:
            raise ValueError("Recommendation not loaded.")

        if self._recommendation.name is None:
            raise ValueError("Recommendation has no name.")

        return self._recommendation.name

    @property
    def title(self) -> str:
        """
        Return the name of the recommendation.
        """
        if self._recommendation is None:
            raise ValueError("Recommendation not loaded.")

        if self._recommendation.title is None:
            raise ValueError("Recommendation has no title.")

        return self._recommendation.title

    @property
    def url(self) -> str:
        """Canonical URL of the recommendation."""
        if self._recommendation is None:
            raise ValueError("Recommendation not loaded.")

        if self._recommendation.url is None:
            raise ValueError("Recommendation has no URL.")

        return self._recommendation.url

    @property
    def version(self) -> str:
        """Version of the recommendation."""
        if self._recommendation is None:
            raise ValueError("Recommendation not loaded.")

        if self._recommendation.version is None:
            raise ValueError("Recommendation has no version.")

        return self._recommendation.version

    def plans(self) -> list["RecommendationPlan"]:
        """
        Return the recommendation plans of this recommendation (i.e. the individual,
        non-overlapping parts or steps of the recommendation).
        """
        return self._recommendation_plans

    def load(self, url: str) -> None:
        """
        Load the recommendation from the FHIR server.
        """
        self._recommendation = self.fetch_recommendation(url)

        for rec_action in self._recommendation.action:
            rec_plan = RecommendationPlan(rec_action.definitionCanonical, self.fhir)
            self._recommendation_plans.append(rec_plan)

        logging.info("Recommendation loaded.")

    def fetch_recommendation(self, canonical_url: str) -> PlanDefinition:
        """
        Fetch the recommendation specified by the canonical URL from the FHIR server.

        This method checks if the PlanDefinition resource referenced by the canonical
        URL is a recommendation (i.e. PlanDefinition.type = #workflow-definition). If
        it is an recommendation-plan instead (i.e. PlanDefinition.type = #eca-rule),
        it will fetch the PlanDefinition that is referenced by the extension[partOf].

        :param canonical_url: Canonical URL of the recommendation
        :return: FHIR PlanDefinition
        """
        rec = self.fhir.fetch_resource("PlanDefinition", canonical_url)
        cc = get_coding(rec.type, CS_PLAN_DEFINITION_TYPE)

        if cc.code == "eca-rule":
            # this is a recommendation-plan resource, try to fetch the actual recommendation
            logging.info("Found recommendation-plan, trying to fetch recommendation.")
            ext = get_extension(rec, EXT_CPG_PARTOF)
            if ext is None:
                raise ValueError(
                    "No partOf extension found in PlanDefinition, can't fetch recommendation."
                )
            assert (
                ext.valueCanonical is not None
            ), "partOf extension has no valueCanonical"

            rec = self.fhir.fetch_resource("PlanDefinition", ext.valueCanonical)
            cc = get_coding(rec.type, CS_PLAN_DEFINITION_TYPE)

        if cc.code != "workflow-definition":
            raise ValueError(f"Unknown recommendation type: {cc.code}")

        return rec


class RecommendationPlan:
    """CPG-on-EBM-on-FHIR RecommendationPlan."""

    class Action:
        """
        Action Definition contained in the PlanDefinition

        Helper class to separate the action definition (contained in PlanDefinition), which specifies the goals,
        type etc., from the activity definition (contained in ActivityDefinition), which may be referenced by the
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
                self._activity = fhir_connector.fetch_resource(
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

        self._recommendation: PlanDefinition | None = None
        self._population: EvidenceVariable | None = None
        self._actions: list[RecommendationPlan.Action] = []
        self._goals: list[PlanDefinitionGoal] = []

        self.load()

    def load(self) -> None:
        """
        Load the recommendation from the FHIR server.
        """

        plan_def = self.fetch_recommendation_plan(self.canonical_url)

        ev = self.fhir.fetch_resource("EvidenceVariable", plan_def.subjectCanonical)

        self._recommendation = plan_def
        self._population = ev
        self._actions = [
            RecommendationPlan.Action(
                action, goals=plan_def.goal, fhir_connector=self.fhir
            )
            for action in plan_def.action
        ]
        self._goals = plan_def.goal

        logging.info(f'Recommendation plan "{plan_def.id}" loaded.')

    def fetch_recommendation_plan(self, canonical_url: str) -> PlanDefinition:
        """
        Fetch the recommendation specified by the canonical URL from the FHIR server.

        This method checks if the PlanDefinition resource referenced by the canonical
        URL is a recommendation (i.e. PlanDefinition.type = #workflow-definition). If
        it is an recommendation-plan instead (i.e. PlanDefinition.type = #eca-rule),
        it will fetch the PlanDefinition that is referenced by the extension[partOf].

        :param canonical_url: Canonical URL of the recommendation
        :return: FHIR PlanDefinition
        """
        rec = self.fhir.fetch_resource("PlanDefinition", canonical_url)
        cc = get_coding(rec.type, CS_PLAN_DEFINITION_TYPE)

        if cc.code != "eca-rule":
            raise ValueError(f"Unknown recommendation type: {cc.code}")

        return rec

    @property
    def name(self) -> str:
        """
        Get the name of the recommendation plan.
        """
        if self._recommendation is None:
            raise ValueError("Recommendation not loaded.")

        if self._recommendation.name is None:
            raise ValueError("Recommendation has no name.")

        return self._recommendation.name

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
