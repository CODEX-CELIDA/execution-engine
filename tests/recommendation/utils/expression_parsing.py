import re
from itertools import product

import pandas as pd
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

CRITERION_COMBINATION_PATTERN = re.compile(r"([\!\?]?)([A-Z][A-Za-z0-9_]+)([<=>]?)")


def criteria_combination_str_to_df(criteria_str: str) -> pd.DataFrame:
    """
    Converts a criteria combination string into a DataFrame.

    The function parses a string of criteria, each represented by their name
    and a comparator, optionally prefixed with '!' for criteria that must
    always be absent or '?' for criteria that are optional. Criteria without a
    prefix are considered always present. It generates all possible combinations
    of optional criteria and creates a DataFrame where each row represents a
    combination, columns represent criteria (named by criterion and comparator),
    and cell values indicate the presence (True) or absence (False) of each criterion.

    Parameters:
    - criteria_str (str): A string of criteria, separated by spaces. Each criterion
      consists of a condition prefix ('!', '?', or none), a criterion name matching
      the pattern ([A-Z][A-Za-z0-9_]+), and a comparator (e.g., '>=', '<=', '=').

    Returns:
    - pd.DataFrame: A DataFrame with a MultiIndex on the columns representing
      criteria names and their comparators, and boolean values indicating the
      presence or absence of each criterion in every combination.

    Raises:
    - ValueError: If any criterion in the string is invalid, either by name or
      because it lacks a comparator.

    Example:
    >>> criteria_str = "A>= !B<= ?C="
    >>> df = criteria_combination_str_to_df(criteria_str)
    >>> df.head()
    """

    split_criteria = []
    for c in criteria_str.split():
        match = CRITERION_COMBINATION_PATTERN.fullmatch(c)
        if not match:
            raise ValueError(f"Invalid criterion name: {c}")

        condition, criterion_name, comparator = match.groups()
        if comparator == "":
            comparator = "="
        elif comparator not in ("<", ">", "="):
            raise ValueError(f"Invalid comparator: {comparator}")

        split_criteria.append((condition, criterion_name, comparator))

    always_present = []
    always_absent = []
    optional = []

    for condition, criterion_name, comparator in split_criteria:
        criterion = (criterion_name, comparator)

        match condition:
            case "!":
                always_absent.append(criterion)
            case "?":
                optional.append(criterion)
            case "":
                always_present.append(criterion)
            case _:
                raise ValueError(f"Invalid condition: {condition}")

    # Generate all combinations for optional criteria
    combinations = list(product([True, False], repeat=len(optional)))

    # Create a DataFrame from combinations
    df = pd.DataFrame(combinations, columns=optional)

    # Add columns for always present and always absent criteria
    for c in always_present:
        df[c] = True
    for c in always_absent:
        df[c] = False

    # Ensure the column order preserves the original criteria sequence (as much as possible)
    criteria_order = [c[1:] for c in split_criteria]
    col_order = sorted(df.columns, key=lambda x: criteria_order.index(x))
    df = df[col_order]

    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["criterion", "comparator"]
    )

    return df


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
