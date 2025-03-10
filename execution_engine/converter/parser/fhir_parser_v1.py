from typing import Callable, Type, cast

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
from execution_engine.util import logic as logic


def characteristic_code_to_expression(
    code: str, threshold: int | None = None
) -> Type[logic.BooleanFunction] | Callable:
    """
    Convert a characteristic combination code (from FHIR) to a criterion combination operator (for OMOP).
    """
    simple_ops = {
        "all-of": logic.And,
        "any-of": logic.Or,
        "all-or-none": logic.AllOrNone,
    }

    count_ops = {
        "at-least": logic.MinCount,
        "at-most": logic.MaxCount,
        "exactly": logic.ExactCount,
        "all-or-none": logic.AllOrNone,
    }

    if code in simple_ops:
        return simple_ops[code]

    if code in count_ops:
        if threshold is None:
            raise ValueError(f"Threshold must be set for operator {code}")
        return lambda *args, category: count_ops[code](*args, threshold=threshold)

    raise NotImplementedError(f'Combination "{code}" not implemented')


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
        combo: logic.BooleanFunction,
    ) -> logic.BooleanFunction:
        """
        Parses the timeFromEvent elements and updates the logic.BooleanFunction.

        Root element timeFromEvent specifies the time within which the population is valid.
        Non-root elements act as filters for the criteria they are attached to, meaning only criteria within the timeFromEvent timeframe will be observed to determine if the patient is part of the population.

        Args:
            tfes (list[EvidenceVariableCharacteristicTimeFromEvent]): List of timeFromEvent elements.
            combo (logic.BooleanFunction): The criterion combination to update.

        Returns:
            TemporalIndicatorCombination: Updated criterion combination.
        """
        if len(tfes) != 1:
            raise ValueError(f"Expected exactly 1 timeFromEvent, got {len(tfes)}")

        tfe = tfes[0]

        converter = self.time_from_event_converters.get(tfe)

        new_combo = converter.to_temporal_combination(combo)

        if not isinstance(new_combo, logic.BooleanFunction):
            raise ValueError(f"Expected BooleanFunction, got {type(new_combo)}")

        return new_combo

    def parse_characteristics(self, ev: EvidenceVariable) -> logic.BooleanFunction:
        """
        Parses the EvidenceVariable characteristics and returns either a BooleanFunction.

        Root element timeFromEvent specifies the time within which the population is valid.
        Non-root elements act as filters for the criteria they are attached to, meaning only criteria within the
        timeFromEvent timeframe will be observed to determine if the patient is part of the population.

        Args:
            ev (EvidenceVariable): The evidence variable to parse.

        Returns:
            BooleanFunction: The parsed criterion combination.
        """

        def build_criterion(
            characteristic: EvidenceVariableCharacteristic, is_root: bool
        ) -> logic.BaseExpr:
            """
            Recursively build Symbol or BooleanFunction from a single
            EvidenceVariableCharacteristic.

            Args:
                characteristic (EvidenceVariableCharacteristic): The characteristic to build from.
                is_root (bool): Indicates if the characteristic is the root element.

            Returns:
                Union[Symbol, BooleanFunction]: The built criterion or criterion combination.
            """

            combo: logic.BooleanFunction

            # If this characteristic is itself a combination
            if characteristic.definitionByCombination is not None:
                expr = characteristic_code_to_expression(
                    characteristic.definitionByCombination.code,
                    threshold=None,  # or parse an actual threshold if needed
                )

                children = []

                for sub_char in characteristic.definitionByCombination.characteristic:
                    children.append(build_criterion(sub_char, is_root=False))

                combo = expr(*children)

                if characteristic.exclude:
                    combo = logic.Not(
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
            crit = converter.to_expression()

            if not isinstance(crit, logic.BaseExpr):
                raise ValueError(f"Expected BaseExpr, got {type(crit)}")

            return crit

        # Top-level: if there's exactly one characteristic and it's a combination, just parse it directly.

        if (
            len(ev.characteristic) == 1
            and ev.characteristic[0].definitionByCombination is not None
        ):
            combo = build_criterion(ev.characteristic[0], is_root=True)
            assert isinstance(combo, logic.BooleanFunction)
            return combo

        # Otherwise, gather them under an ALL_OF (AND) combination by default:
        children = []

        for c in ev.characteristic:
            children.append(build_criterion(c, is_root=True))

        combo = logic.And(*children)

        return combo

    def parse_actions(
        self,
        actions_def: list[fhir.RecommendationPlan.Action],
        rec_plan: fhir.RecommendationPlan,
    ) -> logic.BooleanFunction:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """

        def action_to_combination(
            actions_def: list[fhir.RecommendationPlan.Action],
            parent: fhir.RecommendationPlan | fhir.RecommendationPlan.Action,
        ) -> logic.BooleanFunction:
            # loop through PlanDefinition.action elements and find the corresponding Action object (by action.code)
            actions: list[logic.BaseExpr] = []

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

                    actions.append(action_conv.to_expression())

            action_combination_expr = self.parse_action_combination_method(
                parent.fhir()
            )

            for action_criterion in actions:
                if not isinstance(
                    action_criterion, (logic.Symbol, logic.BooleanFunction)
                ):
                    raise ValueError(f"Invalid action type: {type(action_criterion)}")

            return action_combination_expr(*actions)

        return action_to_combination(actions_def, rec_plan)

    def parse_action_combination_method(
        self, action_parent: PlanDefinition | PlanDefinitionAction
    ) -> Type[logic.BooleanFunction] | Callable:
        """
        Get the correct action combination based on the action selection behavior.
        """

        selection_behaviors = [
            action.selectionBehavior for action in action_parent.action
        ]
        if not len(set(selection_behaviors)) == 1:
            raise ValueError("All actions must have the same selection behaviour.")

        selection_behavior = selection_behaviors[0]

        expr: Type[logic.BooleanFunction] | Callable

        match selection_behavior:
            case "any" | "one-or-more":
                expr = logic.Or
            case "all":
                expr = logic.And
            case "all-or-none":
                expr = logic.AllOrNone
            case "exactly-one":
                expr = lambda *args: logic.ExactCount(*args, threshold=1)
            case "at-most-one":
                expr = lambda *args: logic.MaxCount(*args, threshold=1)
            case _:
                raise NotImplementedError(
                    f"Selection behavior {selection_behavior} not implemented."
                )

        return expr
