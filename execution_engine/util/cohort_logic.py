from typing import Any

import sympy

from execution_engine.constants import CohortCategory


class Expr(sympy.Expr):
    """
    An Expr object represents a symbolic expression.
    Extended to include a cohort_category attribute.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Expr":
        """
        Create a new Expr object.

        :param args: The arguments.
        :param kwargs: The keyword arguments (must contain cohort_category).
        :return: A new Expr object.
        """
        obj = sympy.Expr.__new__(cls, *args)
        obj.cohort_category = kwargs.get("cohort_category")
        return obj


class Symbol(sympy.Symbol):
    """
    A Symbol object represents a symbol in a symbolic expression.
    Extended to include a cohort_category attribute.
    """

    def __new__(cls, name: str, **kwargs: Any) -> "Symbol":
        """
        Create a new Symbol object.

        :param name: The name of the symbol.
        :param kwargs: The keyword arguments (must contain criterion).
        :return: A new Symbol object.
        """
        obj = sympy.Symbol.__new__(cls, name)
        obj.criterion = kwargs.get("criterion")
        return obj

    @property
    def cohort_category(self) -> CohortCategory:
        """
        Get the cohort category of the symbol.

        :return: The cohort category.
        """
        return self.criterion.cohort_category


class And(sympy.And):
    """
    A And object represents a logical AND operation.
    Extended to include a cohort_category attribute.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "And":
        """
        Create a new And object.

        :param args: The arguments.
        :param kwargs: The keyword arguments (must contain cohort_category).
        :return: A new And object.
        """
        obj = sympy.And.__new__(cls, *args)
        obj.cohort_category = kwargs.get("cohort_category")
        return obj


class Or(sympy.Or):
    """
    A Or object represents a logical OR operation.
    Extended to include a cohort_category attribute.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Or":
        """
        Create a new Or object.

        :param args: The arguments.
        :param kwargs: The keyword arguments (must contain cohort_category).
        :return: A new Or object.
        """
        obj = sympy.Or.__new__(cls, *args)
        obj.cohort_category = kwargs.get("cohort_category")
        return obj


class Not(sympy.Not):
    """
    A Not object represents a logical NOT operation.
    Extended to include a cohort_category attribute.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Not":
        """
        Create a new Not object.

        :param args: The arguments.
        :param kwargs: The keyword arguments (must contain cohort_category).
        :return: A new Not object.
        """
        obj = sympy.Not.__new__(cls, *args)
        obj.cohort_category = kwargs.get("cohort_category")
        return obj
