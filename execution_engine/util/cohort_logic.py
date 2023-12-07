from typing import Any, Generic, Type, TypeVar

import sympy

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion

T = TypeVar("T", bound="CohortCategorized")


class CohortCategorized(sympy.Basic, Generic[T]):
    """
    A base class for cohort categorized symbolic objects.
    """

    category: CohortCategory

    def __new__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """
        Create a new CohortCategorized object.

        :param args: The arguments.
        :param kwargs: The keyword arguments (must contain category).
        :return: A new CohortCategorized object.
        """
        obj = super().__new__(cls, *args)
        if not type(obj) == cls:
            # The sympy.And, sympy.Or classes return the original symbol if only one argument is given
            return obj

        category = kwargs.get("category")
        assert category is not None, "Category must be set"
        assert isinstance(
            category, CohortCategory
        ), "Category must be a CohortCategory object"

        obj.category = category

        return obj


class Expr(CohortCategorized, sympy.Expr):
    """
    An Expr object represents a symbolic expression.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class And(CohortCategorized, sympy.And):
    """
    A And object represents a logical AND operation.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Or(CohortCategorized, sympy.Or):
    """
    A Or object represents a logical OR operation.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Not(CohortCategorized, sympy.Not):
    """
    A Not object represents a logical NOT operation.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Symbol(sympy.Symbol):
    """
    A Symbol object represents a symbol in a symbolic expression.
    Extended to include a criterion and a category attribute.

    :param name: The name of the symbol.
    :param kwargs: The keyword arguments (must contain criterion).
    """

    criterion: Criterion

    def __new__(cls, name: str, **kwargs: Any) -> "Symbol":
        """
        Create a new Symbol object.

        :param name: The name of the symbol.
        :param kwargs: The keyword arguments (must contain criterion).
        :return: A new Symbol object.
        """
        obj = super().__new__(cls, name)
        obj.criterion = kwargs.get("criterion")
        assert obj.criterion is not None, "Criterion must be set"
        assert isinstance(
            obj.criterion, Criterion
        ), "Criterion must be a Criterion object"
        return obj

    @property
    def category(self) -> CohortCategory:
        """
        Get the cohort category of the symbol.

        :return: The cohort category.
        """
        return self.criterion.category
