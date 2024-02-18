from typing import Tuple, Union, cast

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)

from execution_engine import fhir
from execution_engine.constants import CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.ventilator_management import (
    VentilatorManagementAction,
)
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
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
from execution_engine.converter.converter import (
    CriterionConverter,
    CriterionConverterFactory,
)
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.goal.laboratory_value import LaboratoryValueGoal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)
from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir_omop_mapping import (
    ActionSelectionBehavior,
    characteristic_to_criterion,
)
from execution_engine.omop import cohort
from execution_engine.omop.cohort import PopulationInterventionPair
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod


class FhirParser:
    """
    A class for parsing FHIR resources into OMOP cohort objects.

    This class provides methods for parsing FHIR resources, such as Recommendation (PlanDefinition) and EvidenceVariable,
    into OMOP cohort objects, such as PopulationInterventionPair and Recommendation. The parsed objects include
    characteristics, actions, goals, and other relevant metadata.
    """

    @staticmethod
    def _init_characteristics_factory() -> CriterionConverterFactory:
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
    def _init_action_factory() -> CriterionConverterFactory:
        af = CriterionConverterFactory()
        af.register(DrugAdministrationAction)
        af.register(VentilatorManagementAction)
        af.register(BodyPositioningAction)

        return af

    @staticmethod
    def _init_goal_factory() -> CriterionConverterFactory:
        gf = CriterionConverterFactory()
        gf.register(LaboratoryValueGoal)
        gf.register(VentilatorManagementGoal)

        return gf

    def _parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        cf = self._init_characteristics_factory()

        def get_characteristic_combination(
            characteristic: EvidenceVariableCharacteristic,
        ) -> Tuple[CharacteristicCombination, EvidenceVariableCharacteristic]:
            comb = CharacteristicCombination(
                CharacteristicCombination.Code(
                    characteristic.definitionByCombination.code
                ),
                exclude=characteristic.exclude,
            )
            characteristics = characteristic.definitionByCombination.characteristic
            return comb, characteristics

        def get_characteristics(
            comb: CharacteristicCombination,
            characteristics: list[EvidenceVariableCharacteristic],
        ) -> CharacteristicCombination:
            sub: Union[CriterionConverter, CharacteristicCombination]
            for c in characteristics:
                if c.definitionByCombination is not None:
                    sub = get_characteristics(*get_characteristic_combination(c))
                else:
                    sub = cf.get(c)
                    sub = cast(
                        AbstractCharacteristic, sub
                    )  # only for mypy, doesn't do anything at runtime
                comb.add(sub)

            return comb

        if len(
            ev.characteristic
        ) == 1 and fhir.RecommendationPlan.is_combination_definition(
            ev.characteristic[0]
        ):
            comb, characteristics = get_characteristic_combination(ev.characteristic[0])
        else:
            comb = CharacteristicCombination(
                CharacteristicCombination.Code.ALL_OF, exclude=False
            )
            characteristics = ev.characteristic

        get_characteristics(comb, characteristics)

        return comb

    def _parse_actions(
        self, actions_def: list[fhir.RecommendationPlan.Action]
    ) -> Tuple[list[AbstractAction], ActionSelectionBehavior]:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """

        af = self._init_action_factory()
        gf = self._init_goal_factory()

        assert (
            len(set([a.action.selectionBehavior for a in actions_def])) == 1
        ), "All actions must have the same selection behaviour."

        selection_behavior = ActionSelectionBehavior(
            actions_def[0].action.selectionBehavior
        )

        # loop through PlanDefinition.action elements and find the corresponding Action object (by action.code)
        actions: list[AbstractAction] = []
        for action_def in actions_def:
            action = af.get(action_def)
            action = cast(
                AbstractAction, action
            )  # only for mypy, doesn't do anything at runtime

            for goal_def in action_def.goals:
                goal = gf.get(goal_def)
                goal = cast(Goal, goal)
                action.goals.append(goal)

            actions.append(action)

        return actions, selection_behavior

    def _action_combination(
        self, selection_behavior: ActionSelectionBehavior
    ) -> CriterionCombination:
        """
        Get the correct action combination based on the action selection behavior.
        """

        if selection_behavior.code == CharacteristicCombination.Code.ANY_OF:
            operator = CriterionCombination.Operator("OR")
        elif selection_behavior.code == CharacteristicCombination.Code.ALL_OF:
            operator = CriterionCombination.Operator("AND")
        elif selection_behavior.code == CharacteristicCombination.Code.AT_LEAST:
            if selection_behavior.threshold == 1:
                operator = CriterionCombination.Operator("OR")
            else:
                operator = CriterionCombination.Operator(
                    "AT_LEAST", threshold=selection_behavior.threshold
                )
        elif selection_behavior.code == CharacteristicCombination.Code.AT_MOST:
            operator = CriterionCombination.Operator(
                "AT_MOST", threshold=selection_behavior.threshold
            )
        elif selection_behavior.code == CharacteristicCombination.Code.EXACTLY:
            operator = CriterionCombination.Operator(
                "EXACTLY", threshold=selection_behavior.threshold
            )
        elif selection_behavior.code == CharacteristicCombination.Code.ALL_OR_NONE:
            operator = CriterionCombination.Operator("ALL_OR_NONE")
        else:
            raise NotImplementedError(
                f"Selection behavior {str(selection_behavior.code)} not implemented."
            )
        return CriterionCombination(
            name="intervention_actions",
            category=CohortCategory.INTERVENTION,
            exclude=False,
            operator=operator,
        )

    def parse_recommendation_from_url(
        self, url: str, package_version: str, fhir_client: FHIRClient
    ) -> cohort.Recommendation:
        """
        Creates a Recommendation object by fetching and parsing recommendation data from a given URL.

        This function utilizes a FHIR connector to access recommendation data, constructs population and intervention
        pairs based on the recommendation plans, and aggregates them into a Recommendation object which includes
        metadata like name, version, and description.

        Args:
            url (str): The URL from which to fetch the recommendation data.
            package_version (str): The version of the recommendation package to be used.
            fhir_client (FHIRClient): An instance of FHIRClient used to connect to and fetch data from a FHIR server.

        Returns:
            cohort.Recommendation: An instance of the Recommendation class populated with the parsed recommendation data,
            including population intervention pairs and other relevant metadata.

        Raises:
            ValueError: If an action within a recommendation plan is None, indicating incomplete or invalid data.
        """
        rec = fhir.Recommendation(
            url,
            package_version=package_version,
            fhir_connector=fhir_client,
        )

        pi_pairs: list[PopulationInterventionPair] = []

        base_criterion = PatientsActiveDuringPeriod(name="active_patients")

        for rec_plan in rec.plans():
            pi_pair = PopulationInterventionPair(
                name=rec_plan.name,
                url=rec_plan.url,
                base_criterion=base_criterion,
            )

            # parse population and create criteria
            characteristics = self._parse_characteristics(rec_plan.population)

            for characteristic in characteristics:
                pi_pair.add_population(characteristic_to_criterion(characteristic))

            # parse intervention and create criteria
            actions, selection_behavior = self._parse_actions(rec_plan.actions)
            comb_actions = self._action_combination(selection_behavior)

            for action in actions:
                if action is None:
                    raise ValueError("Action is None.")
                comb_actions.add(action.to_criterion())

            pi_pair.add_intervention(comb_actions)

            pi_pairs.append(pi_pair)

        recommendation = cohort.Recommendation(
            pi_pairs,
            base_criterion=base_criterion,
            url=rec.url,
            name=rec.name,
            title=rec.title,
            version=rec.version,
            description=rec.description,
            package_version=rec.package_version,
        )

        return recommendation
