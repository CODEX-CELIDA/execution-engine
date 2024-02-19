from typing import Tuple, Union, cast

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

from execution_engine import fhir
from execution_engine.constants import CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.characteristic.combination import (
    CharacteristicCombination,
)
from execution_engine.converter.converter import CriterionConverter
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.parser.base import FhirParserInterface
from execution_engine.fhir_omop_mapping import ActionSelectionBehavior
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination


class FhirParserV1(FhirParserInterface):
    """
    Parses a recommendation in CPG-on-EBMonFHIR format before version v0.8.

    Before version 0.8, the method of combination of actions was read from the PlanDefinition.action.selectionBehavior
    element. In newer versions, this has been replaced by an extension to PlanDefinition.

    Also, before version 0.8, only a single level of actions could be read (no nested actions with different
    combination methods).
    """

    def parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        cf = self.init_characteristics_factory()

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

    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> CriterionCombination:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """
        af = self.init_action_factory()
        gf = self.init_goal_factory()

        def action_to_combination(
            actions_def: list[fhir.RecommendationPlan.Action],
            parent: fhir.RecommendationPlan | fhir.RecommendationPlan.Action,
        ) -> CriterionCombination:
            # loop through PlanDefinition.action elements and find the corresponding Action object (by action.code)
            actions: list[Criterion | CriterionCombination] = []

            for action_def in actions_def:
                # check if is combination of actions
                if action_def.nested_actions:
                    # todo: make sure code is used correctly and we don't have a definition?
                    action_combination = action_to_combination(
                        action_def.nested_actions, action_def.fhir()
                    )
                    actions.append(action_combination)
                else:
                    action_conv = af.get(action_def)
                    action_conv = cast(
                        AbstractAction, action_conv
                    )  # only for mypy, doesn't do anything at runtime

                    for goal_def in action_def.goals_fhir:
                        goal = gf.get(goal_def)
                        goal = cast(Goal, goal)
                        action_conv.goals.append(goal)

                    actions.append(action_conv.to_criterion())

            action_combination = self.parse_action_combination_method(parent.fhir())

            for action_criterion in actions:
                if not isinstance(action_criterion, (CriterionCombination, Criterion)):
                    raise ValueError(f"Invalid action type: {type(action_criterion)}")

                action_combination.add(action_criterion)

            return action_combination

        return action_to_combination(actions_def, rec_plan)

    def parse_action_combination_method(
        self, action_parent: PlanDefinition | PlanDefinitionAction
    ) -> CriterionCombination:
        """
        Get the correct action combination based on the action selection behavior.
        """

        selection_behaviors = [
            action.selectionBehavior for action in action_parent.action
        ]
        if not len(set(selection_behaviors)) == 1:
            raise ValueError("All actions must have the same selection behaviour.")

        selection_behavior = ActionSelectionBehavior(selection_behaviors[0])

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
