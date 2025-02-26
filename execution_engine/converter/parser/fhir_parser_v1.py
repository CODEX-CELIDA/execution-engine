from typing import Union, cast

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicTimeFromEvent,
)
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

from execution_engine import fhir
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.omop.criterion.abstract import AbstractCriterion, Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)


def characteristic_code_to_criterion_combination_operator(
    code: str, threshold: int | None = None
) -> LogicalCriterionCombination.Operator:
    """
    Convert a characteristic combination code (from FHIR) to a criterion combination operator (for OMOP).
    """
    mapping = {
        "all-of": LogicalCriterionCombination.Operator.AND,
        "any-of": LogicalCriterionCombination.Operator.OR,
        "at-least": LogicalCriterionCombination.Operator.AT_LEAST,
        "at-most": LogicalCriterionCombination.Operator.AT_MOST,
        "exactly": LogicalCriterionCombination.Operator.EXACTLY,
        "all-or-none": LogicalCriterionCombination.Operator.ALL_OR_NONE,
    }

    if code not in mapping:
        raise NotImplementedError(f"Unknown combination code: {code}")
    return LogicalCriterionCombination.Operator(
        operator=mapping[code], threshold=threshold
    )


class FhirRecommendationParserV1(FhirRecommendationParserInterface):
    """
    Parses a recommendation in CPG-on-EBMonFHIR format before version v0.8.

    Before version 0.8, the method of combination of actions was read from the PlanDefinition.action.selectionBehavior
    element. In newer versions, this has been replaced by an extension to PlanDefinition.

    Also, before version 0.8, only a single level of actions could be read (no nested actions with different
    combination methods).
    """

    def parse_time_from_event(
        self,
        tfes: list[EvidenceVariableCharacteristicTimeFromEvent],
        combo: CriterionCombination,
    ) -> CriterionCombination:
        """
        Parses the timeFromEvent elements and updates the CriterionCombination.

        Root element timeFromEvent specifies the time within which the population is valid.
        Non-root elements act as filters for the criteria they are attached to, meaning only criteria within the timeFromEvent timeframe will be observed to determine if the patient is part of the population.

        Args:
            tfes (list[EvidenceVariableCharacteristicTimeFromEvent]): List of timeFromEvent elements.
            combo (CriterionCombination): The criterion combination to update.

        Returns:
            TemporalIndicatorCombination: Updated criterion combination.
        """
        if len(tfes) != 1:
            raise ValueError(f"Expected exactly 1 timeFromEvent, got {len(tfes)}")

        tfe = tfes[0]

        converter = self.time_from_event_converters.get(tfe)

        new_combo = converter.to_temporal_combination(combo)

        return new_combo

    def parse_characteristics(self, ev: EvidenceVariable) -> CriterionCombination:
        """
        Parses the EvidenceVariable characteristics and returns either a single Criterion
        or a CriterionCombination.

        Root element timeFromEvent specifies the time within which the population is valid.
        Non-root elements act as filters for the criteria they are attached to, meaning only criteria within the timeFromEvent timeframe will be observed to determine if the patient is part of the population.

        Args:
            ev (EvidenceVariable): The evidence variable to parse.

        Returns:
            CriterionCombination: The parsed criterion combination.
        """

        def build_criterion(
            characteristic: EvidenceVariableCharacteristic, is_root: bool
        ) -> Union[Criterion, CriterionCombination]:
            """
            Recursively build Criterion or CriterionCombination from a single
            EvidenceVariableCharacteristic.

            Args:
                characteristic (EvidenceVariableCharacteristic): The characteristic to build from.
                is_root (bool): Indicates if the characteristic is the root element.

            Returns:
                Union[Criterion, CriterionCombination]: The built criterion or criterion combination.
            """

            combo: CriterionCombination

            # If this characteristic is itself a combination
            if characteristic.definitionByCombination is not None:
                operator = characteristic_code_to_criterion_combination_operator(
                    characteristic.definitionByCombination.code,
                    threshold=None,  # or parse an actual threshold if needed
                )
                combo = LogicalCriterionCombination(
                    operator=operator,
                )

                for sub_char in characteristic.definitionByCombination.characteristic:
                    combo.add(build_criterion(sub_char, is_root=False))

                if characteristic.exclude:
                    combo = LogicalCriterionCombination.Not(
                        combo,
                    )

                if characteristic.timeFromEvent is not None:
                    combo = self.parse_time_from_event(
                        characteristic.timeFromEvent, combo
                    )

                return combo

            # Else it's a single characteristic
            converter = self.characteristics_converters.get(characteristic)
            converter = cast(AbstractCharacteristic, converter)
            crit = converter.to_criterion()

            if not isinstance(crit, AbstractCriterion):
                raise ValueError(f"Expected AbstractCriterion, got {type(crit)}")

            return crit

        # Top-level: if there's exactly one characteristic and it's a combination, just parse it directly.

        if (
            len(ev.characteristic) == 1
            and ev.characteristic[0].definitionByCombination is not None
        ):
            combo = build_criterion(ev.characteristic[0], is_root=True)
            assert isinstance(combo, CriterionCombination)
            return combo

        # Otherwise, gather them under an ALL_OF (AND) combination by default:
        combo = LogicalCriterionCombination(
            operator=LogicalCriterionCombination.Operator(
                LogicalCriterionCombination.Operator.AND
            ),
        )
        for c in ev.characteristic:
            combo.add(build_criterion(c, is_root=True))

        return combo

    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> CriterionCombination:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """

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
                        action_def.nested_actions, action_def
                    )
                    actions.append(action_combination)
                else:
                    action_conv = self.action_converters.get(action_def)
                    action_conv = cast(
                        AbstractAction, action_conv
                    )  # only for mypy, doesn't do anything at runtime

                    for goal_def in action_def.goals_fhir:
                        goal = self.goal_converters.get(goal_def)
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

        selection_behavior = selection_behaviors[0]

        match selection_behavior:
            case "any" | "one-or-more":
                operator = LogicalCriterionCombination.Operator("OR")
            case "all":
                operator = LogicalCriterionCombination.Operator("AND")
            case "all-or-none":
                operator = LogicalCriterionCombination.Operator("ALL_OR_NONE")
            case "exactly-one":
                operator = LogicalCriterionCombination.Operator("EXACTLY", threshold=1)
            case "at-most-one":
                operator = LogicalCriterionCombination.Operator("AT_MOST", threshold=1)
            case _:
                raise NotImplementedError(
                    f"Selection behavior {selection_behavior} not implemented."
                )

        return LogicalCriterionCombination(
            operator=operator,
        )
