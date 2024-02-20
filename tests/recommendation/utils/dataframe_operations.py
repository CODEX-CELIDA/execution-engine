from functools import reduce

import pandas as pd
from numpy import typing as npt

from execution_engine.util.interval import IntervalType
from tests.recommendation.utils.expression_parsing import (
    Add,
    And,
    Eq,
    Not,
    Or,
    Symbol,
    parse_expression,
    restore_criterion_name,
)


def elementwise_mask(s1, mask, fill_value=False):
    return s1.combine(mask, lambda x, y: x if y else fill_value)


def elementwise_and(s1, s2):
    return s1.combine(s2, lambda x, y: x & y)


def elementwise_or(s1, s2):
    return s1.combine(s2, lambda x, y: x | y)


def elementwise_add(s1, s2):
    return s1.combine(s2, lambda x, y: x + y)


def elementwise_not(s1):
    return s1.map(lambda x: ~x)


def combine_dataframe_via_logical_expression(
    df: pd.DataFrame, expression: str
) -> npt.NDArray:
    def eval_expr(expr):
        if isinstance(expr, Symbol):
            return df[restore_criterion_name(expr)]
        elif isinstance(expr, And):
            return reduce(elementwise_and, map(eval_expr, expr.args))
        elif isinstance(expr, Or):
            return reduce(elementwise_or, map(eval_expr, expr.args))
        elif isinstance(expr, Not):
            return elementwise_not(eval_expr(expr.args[0]))
        elif isinstance(expr, Eq):
            expr, value = expr.args[0], int(expr.args[1])
            if isinstance(expr, Add):
                return (
                    reduce(
                        elementwise_add,
                        map(
                            lambda col: df[restore_criterion_name(col)]
                            .astype(bool)
                            .astype(int),
                            expr.args,
                        ),
                    )
                    == value
                ).map({False: IntervalType.NEGATIVE, True: IntervalType.POSITIVE})
            else:
                raise ValueError(f"Unsupported expression: {expr}")
        else:
            raise ValueError(f"Unsupported expression: {expr}")

    parsed_expr = parse_expression(expression)

    return eval_expr(parsed_expr)
