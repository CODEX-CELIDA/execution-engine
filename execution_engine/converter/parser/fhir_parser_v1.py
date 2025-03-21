from typing import Callable, Type, cast

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicTimeFromEvent,
)
from fhir.resources.extension import Extension
from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionAction

from execution_engine import fhir
from execution_engine.constants import EXT_RELATIVE_TIME
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.converter.parser.util import wrap_criteria_with_temporal_indicator
from execution_engine.fhir.util import get_extensions, pop_extensions
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
    ) -> list[logic.BaseExpr]:
        """
        Parses `timeFromEvent` elements and converts them into interval-based logical criteria.

        The `timeFromEvent` elements specify time intervals that constrain when the associated
        criteria must occur. These intervals serve as temporal filters, ensuring that only criteria
        occurring within the defined time window contribute to determining population membership.

        This function processes each `timeFromEvent` element by converting it into an interval-based
        logical criterion. These criteria are later used to enforce temporal constraints on individual
        criteria within a characteristic.

        Args:
            tfes (list[EvidenceVariableCharacteristicTimeFromEvent]):
                A list of `timeFromEvent` elements defining temporal constraints.

        Returns:
            list[logic.BaseExpr]:
                A list of interval-based logical expressions representing the extracted temporal constraints.

        Raises:
            ValueError:
                If any `timeFromEvent` element does not yield a valid `logic.BaseExpr`.

        Notes:
            - Each `timeFromEvent` element is processed using a registered converter that transforms
              it into an interval-based logical expression.
            - These interval-based criteria are meant to be combined later (typically with AND) and
              applied to individual criteria rather than an entire combination.
        """
        interval_criteria = []

        for tfe in tfes:
            converter = self.time_from_event_converters.get(tfe)

            interval_criterion = converter.to_interval_criterion()

            if not isinstance(interval_criterion, logic.BaseExpr):
                raise ValueError(
                    f"Expected instance of BaseExpr, got {type(interval_criterion)}"
                )

            interval_criteria.append(interval_criterion)

        return interval_criteria

    def parse_relative_time(
        self,
        relative_time: list[Extension],
    ) -> list[logic.BaseExpr]:
        """
        Parses `extension[relativeTime]` elements and converts them into interval-based logical criteria.

        The `extension[relativeTime]` elements specify time intervals that constrain when the associated
        criteria must occur. These intervals serve as temporal filters, ensuring that only criteria
        occurring within the defined time window contribute to determining population membership.

        This function processes each `extension[relativeTime]` element by converting it into an interval-based
        logical criterion. These criteria are later used to enforce temporal constraints on individual
        criteria within a characteristic.

        Args:
            relative_time (list[Extension]):
                A list of `extension[relativeTime]` elements defining temporal constraints.

        Returns:
            list[logic.BaseExpr]:
                A list of interval-based logical expressions representing the extracted temporal constraints.

        Raises:
            ValueError:
                If any `extension[relativeTime]` element does not yield a valid `logic.BaseExpr`.

        Notes:
            - Each `extension[relativeTime]` element is processed using a registered converter that transforms
              it into an interval-based logical expression.
            - These interval-based criteria are meant to be combined later (typically with AND) and
              applied to individual criteria rather than an entire combination.
        """
        interval_criteria = []

        for ext in relative_time:

            converter = self.relative_time_converters.get(ext)

            interval_criterion = converter.to_interval_criterion()

            if not isinstance(interval_criterion, logic.BaseExpr):
                raise ValueError(
                    f"Expected instance of BaseExpr, got {type(interval_criterion)}"
                )

            interval_criteria.append(interval_criterion)

        return interval_criteria

    def parse_timing(
        self, characteristic: EvidenceVariableCharacteristic, expr: logic.BaseExpr
    ) -> logic.BaseExpr:
        """
        Applies temporal constraints to a given criterion expression based on `timeFromEvent` and
        the relativeTime extension elements.

        This function aggregates all applicable temporal constraints associated with a characteristic,
        including `timeFromEvent` elements and relative timing extensions. These constraints define
        the time intervals within which the given criterion must be evaluated.

        The extracted interval criteria are AND-combined and used to wrap **each individual criterion**
        rather than the entire criterion combination. This prevents unintended constraints that could
        arise if multiple criteria were required to occur simultaneously.

        Args:
            characteristic (EvidenceVariableCharacteristic):
                The characteristic whose timing constraints should be applied.
            expr (logic.BaseExpr):
                The logical expression representing the criterion to be updated.

        Returns:
            logic.BaseExpr:
                The criterion expression with the applied temporal constraints.

        Notes:
            - If `timeFromEvent` elements are present, they are processed using `parse_time_from_event`.
            - If a relative timing extension is present, it is processed separately.
            - All extracted interval criteria are AND-combined and used to wrap each individual
              criterion within the logical expression to ensure correct temporal evaluation.
        """
        interval_criteria: list[logic.BaseExpr] = []

        if characteristic.timeFromEvent is not None:
            interval_criteria.extend(
                self.parse_time_from_event(characteristic.timeFromEvent)
            )

        relative_time = get_extensions(characteristic, EXT_RELATIVE_TIME)

        if relative_time:
            interval_criteria.extend(self.parse_relative_time(relative_time))

        if interval_criteria:
            interval_criterion_combo = logic.And(*interval_criteria)

            expr = wrap_criteria_with_temporal_indicator(expr, interval_criterion_combo)

        return expr

    def process_action_relative_time(
        self, action_def: fhir.RecommendationPlan.Action
    ) -> list[logic.BaseExpr]:
        """
        Processes the relativeTime extension in the given Action and removes it from
        the ActivityDefinition's timing field so that subsequent calls (e.g. process_timing)
        do not re-process it.

        Args:
            action_def (fhir.RecommendationPlan.Action):
                The FHIR action definition that potentially contains relativeTime extensions
                in its ActivityDefinition.

        Returns:
            list[logic.BaseExpr]: A list of expressions derived from the relativeTime extensions,
            or an empty list if none were found.
        """
        if (
            action_def.activity_definition_fhir is None
            or action_def.activity_definition_fhir.timingTiming is None
        ):
            return []

        relative_time = pop_extensions(
            action_def.activity_definition_fhir.timingTiming, EXT_RELATIVE_TIME
        )

        if not relative_time:
            return []

        return self.parse_relative_time(relative_time)

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

            combo: logic.BaseExpr

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

                combo = self.parse_timing(characteristic, combo)

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
                    action_combination = action_to_combination(
                        action_def.nested_actions, action_def
                    )
                    actions.append(action_combination)
                else:

                    # first process a possible relativeTime extension in timingTiming
                    # if there is one, a corresponding logic.Presence (=logic.TemporalMinCount(*args, threshold=1))
                    # expression is constructed and later wrapped around the actual criterion
                    interval_criteria = self.process_action_relative_time(action_def)

                    action_conv = self.action_converters.get(action_def)
                    action_conv = cast(
                        AbstractAction, action_conv
                    )  # only for mypy, doesn't do anything at runtime

                    for goal_def in action_def.goals_fhir:
                        goal = self.goal_converters.get(goal_def)
                        goal = cast(Goal, goal)
                        action_conv.goals.append(goal)

                    expr = action_conv.to_expression()

                    if interval_criteria:
                        expr = wrap_criteria_with_temporal_indicator(
                            expr, logic.And(*interval_criteria)
                        )

                    actions.append(expr)

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
