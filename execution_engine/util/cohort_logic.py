from typing import Any, Generic, Type, TypeVar

import sympy

from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.criterion.abstract import Criterion

T = TypeVar("T", bound="CohortCategorized")


# todo: should NOT_APPLICABLE be included here or not?
class FourValuedLogic(sympy.logic.boolalg.Boolean):
    """
    Defines a four valued logic

    The four values are:
    - POSITIVE
    - NEGATIVE
    - NO_DATA
    - NOT_APPLICABLE

    The logic is defined as follows:

    | left/right | NO_DATA   | POSITIVE | NEGATIVE | NOT_APPLICABLE |
    |------------|----------|----------|----------|----------|
    | NO_DATA     | NO_DATA   | POSITIVE | NEGATIVE | NOT_APPLICABLE |
    | POSITIVE   | POSITIVE | POSITIVE | NEGATIVE | NOT_APPLICABLE |
    | NEGATIVE   | NEGATIVE | NEGATIVE | NEGATIVE | NEGATIVE |
    | NOT_APPLICABLE   | NOT_APPLICABLE | NOT_APPLICABLE | NEGATIVE | NOT_APPLICABLE |
    """

    POSITIVE = sympy.S(IntervalType.POSITIVE)
    NEGATIVE = sympy.S(IntervalType.NEGATIVE)
    NO_DATA = sympy.S(IntervalType.NO_DATA)
    NOT_APPLICABLE = sympy.S(IntervalType.NOT_APPLICABLE)

    def __new__(cls, value: Any) -> "FourValuedLogic":
        """
        Create a new FourValuedLogic object.
        """
        if value in [cls.POSITIVE, True, 1, "True", IntervalType.POSITIVE]:
            return cls.POSITIVE
        elif value in [cls.NEGATIVE, False, 0, "False", IntervalType.NEGATIVE]:
            return cls.NEGATIVE
        elif value in [cls.NO_DATA, "NoData", IntervalType.NO_DATA]:
            return cls.NO_DATA
        elif value in [
            cls.NOT_APPLICABLE,
            "NotApplicable",
            IntervalType.NOT_APPLICABLE,
        ]:
            return cls.NOT_APPLICABLE
        else:
            raise ValueError("Invalid value for FourValuedLogic")

    def __and__(self, other: "FourValuedLogic") -> "FourValuedLogic":
        """Define logic for AND operation"""
        if self == self.NEGATIVE or other == self.NEGATIVE:
            return self.NEGATIVE
        if self == self.NO_DATA and other == self.NO_DATA:
            return self.NO_DATA
        if self == self.NOT_APPLICABLE and other == self.NOT_APPLICABLE:
            return self.NO_DATA
        return self.POSITIVE

    def __or__(self, other: "FourValuedLogic") -> "FourValuedLogic":
        """
        Define logic for OR operation
        """
        if self == self.POSITIVE or other == self.POSITIVE:
            return self.POSITIVE
        if self == self.NO_DATA and other == self.NO_DATA:
            return self.NO_DATA
        return self.NEGATIVE

    def __invert__(self) -> "FourValuedLogic":
        """
        Define logic for NOT operation
        """
        if self == self.POSITIVE:
            return self.NEGATIVE
        if self == self.NEGATIVE:
            return self.POSITIVE
        return self.NO_DATA


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

        params = kwargs.get("params")

        if params is not None:
            assert isinstance(params, dict), "Params must be a dict"
            obj.params = params
        else:
            obj.params = {}

        return obj


class BooleanFunction(CohortCategorized, sympy.logic.boolalg.BooleanFunction):
    """
    A BooleanFunction object represents a boolean function.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Expr(CohortCategorized, sympy.Expr):
    """
    An Expr object represents a symbolic expression.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class NonSimplifiableAnd(BooleanFunction):
    """
    A NonSimplifiableAnd object represents a logical AND operation that cannot be simplified.

    Simplified here means that when this operator is used on a single argument, still this operator is returned
    instead of the argument itself, as is the case with the sympy.And operator.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "NonSimplifiableAnd":
        """
        Create a new NonSimplifiableAnd object.
        """
        if len(args) == 1 and isinstance(args[0], (And, Or)):
            return args[0]

        return super().__new__(cls, *args, **kwargs)


class NoDataPreservingAnd(BooleanFunction):
    """
    A And object represents a logical AND operation.

    This operator handles the three valued algebra of NODATA, POSITIVE, NEGATIVE, according to the following rules:

    | left/right | NODATA   | POSITIVE | NEGATIVE |
    |------------|----------|----------|----------|
    | NODATA     | NODATA   | POSITIVE | NEGATIVE |
    | POSITIVE   | POSITIVE | POSITIVE | NEGATIVE |
    | NEGATIVE   | NEGATIVE | NEGATIVE | NEGATIVE |


    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """

    def doit(self, **hints: Any) -> FourValuedLogic:
        """
        Method to perform the custom operation.
        You can define your custom logic here.
        """
        if len(self.args) == 0:
            raise ValueError("No arguments provided")
        if all(arg == FourValuedLogic.POSITIVE for arg in self.args):
            return FourValuedLogic.POSITIVE
        if any(arg == FourValuedLogic.NEGATIVE for arg in self.args):
            return FourValuedLogic.NEGATIVE
        if any(arg == FourValuedLogic.NO_DATA for arg in self.args):
            return FourValuedLogic.NO_DATA
        raise ValueError("Invalid value for ThreeValuedLogic")


class LeftDependentToggle(BooleanFunction):
    """
    A LeftDependentToggle object represents a logical AND operation if the left operand is positive,
    otherwise it returns NOT_APPLICABLE.

    | left/right | NODATA   | POSITIVE | NEGATIVE |
    |------------|----------|----------|----------|
    | NODATA     | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
    | POSITIVE   | NODATA | POSITIVE | NEGATIVE |
    | NEGATIVE   | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
    """

    def __new__(cls, left: Expr, right: Expr, **kwargs: Any) -> "LeftDependentToggle":
        """
        Create a new LeftDependentToggle object.
        """
        return Expr.__new__(cls, left, right, **kwargs)

    @property
    def left(self) -> Expr:
        """Returns the left operand"""
        return self.args[0]

    @property
    def right(self) -> Expr:
        """Returns the right operand"""
        return self.args[1]

    def doit(self, **hints: Any) -> FourValuedLogic:
        """
        Method to perform the custom operation.
        You can define your custom logic here.
        """
        if self.left in [FourValuedLogic.NEGATIVE, FourValuedLogic.NO_DATA]:
            return FourValuedLogic.NOT_APPLICABLE

        return self.right


class And(BooleanFunction, sympy.And):
    """
    A And object represents a logical AND operation.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Or(BooleanFunction, sympy.Or):
    """
    A Or object represents a logical OR operation.
    Extended to include a category attribute.

    :param args: The arguments.
    :param kwargs: The keyword arguments (must contain category).
    """


class Not(BooleanFunction, sympy.Not):
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

        params = kwargs.get("params")
        if params is not None:
            assert isinstance(params, dict), "Params must be a dict"
            obj.params = params
        else:
            obj.params = {}

        return obj

    @property
    def category(self) -> CohortCategory:
        """
        Get the cohort category of the symbol.

        :return: The cohort category.
        """
        return self.criterion.category
