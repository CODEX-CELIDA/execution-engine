import logging
from typing import Callable, cast

from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.util import logic
from execution_engine.util.temporal_logic_util import Presence


def _wrap_criteria_with_factory(
    expr: logic.BaseExpr,
    factory: Callable[[logic.BaseExpr], logic.TemporalCount],
) -> logic.Expr:
    """
    Recursively wraps all Criterion instances within a combination using the specified factory.

    :param expr: A single Criterion or an expression to be processed.
    :param factory: A callable that takes a Criterion or expression and returns a TemporalCount.
    :return: A new TemporalCount where all Criterion instances have been wrapped using the factory.
    :raises ValueError: If an unexpected element type is encountered.
    """

    from digipod.concepts import OMOP_SURGICAL_PROCEDURE

    new_expr: logic.Expr

    if isinstance(expr, Criterion):
        new_expr = factory(expr)
    elif isinstance(expr, logic.Expr):

        # Create a new combination of the same type with the same operator
        args = []

        # Loop through all elements
        for element in expr.args:
            if isinstance(element, logic.Expr):
                # Recursively wrap nested combinations
                args.append(_wrap_criteria_with_factory(element, factory))
            elif isinstance(element, Criterion):
                # Wrap individual criteria with the factory

                if (
                    isinstance(element, ProcedureOccurrence)
                    and element.concept.concept_id == OMOP_SURGICAL_PROCEDURE
                    and element.concept.vocabulary_id == "SNOMED"
                ):
                    logging.warning(
                        "Removing Surgical Procedure Criterion in TimeFromEvent-SurgicalOperationDate"
                    )
                    continue

                args.append(factory(element))

            else:
                raise ValueError(f"Unexpected element type: {type(element)}")

        new_expr = expr.__class__(*args)
    else:
        raise ValueError(f"Unexpected element type: {type(expr)}")

    return new_expr


def wrap_criteria_with_temporal_indicator(
    expr: logic.BaseExpr,
    interval_criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Wraps all Criterion instances in a combination with a TemporalCount (with interval_criterion).

    :param expr: A single Criterion or an expression to be wrapped.
    :param interval_criterion: A Criterion or expression that defines the temporal interval.
    :return: A new expression where all Criterion instances are wrapped with a TemporalCount (with interval_criterion).
    """
    temporal_combo_factory = lambda criterion: Presence(
        criterion=criterion, interval_criterion=interval_criterion
    )

    new_combo = cast(
        logic.TemporalMinCount,
        _wrap_criteria_with_factory(expr, temporal_combo_factory),
    )

    return new_combo
