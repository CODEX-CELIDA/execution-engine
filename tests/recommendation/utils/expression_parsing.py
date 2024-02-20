import re

import sympy
from sympy import (  # noqa: F401 -- this module should expose these classes
    Add,
    And,
    Eq,
    Expr,
    Not,
    Or,
    Symbol,
)

CRITERION_PATTERN = re.compile(
    r"(?!Eq\b|And\b|Or\b|Not\b|Add\b|Div\b|Sub\b)([A-Z][A-Za-z0-9_]+)([<=>]?)"
)


def _sympify_criteria_names(expression: str) -> str:
    """
    Replace criterion names in the expression with sympy symbols.
    """

    def replace(match):
        criterion_name, comparator = match.groups()
        if comparator == "":
            comparator = "="
        comparator_map = {"<": "lt", ">": "gt", "=": "eq"}
        name = f"__{criterion_name}__{comparator_map[comparator]}__"

        return name

    return CRITERION_PATTERN.sub(replace, expression)


def restore_criterion_name(symbol: sympy.Symbol) -> tuple[str, str]:
    """
    Restore criterion name from the sympy symbol.
    """
    parts = symbol.name.split("__")
    criterion_name = parts[1]
    comparator = parts[2]
    comparator_map = {"lt": "<", "gt": ">", "eq": "="}
    return criterion_name, comparator_map[comparator]


def parse_expression(expression: str) -> Expr:
    """
    Parse the expression into a sympy expression.
    """
    expr = _sympify_criteria_names(expression)
    parsed_expr = sympy.parse_expr(expr)
    return parsed_expr
