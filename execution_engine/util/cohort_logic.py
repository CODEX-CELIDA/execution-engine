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
        if type(obj) is not cls:
            # The sympy.And, sympy.Or classes return the original symbol if only one argument is given
            return obj

        category = kwargs.get("category")
        assert category is not None, "Category must be set"
        assert isinstance(
            category, CohortCategory
        ), "Category must be a CohortCategory object"

        obj.category = category

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

    The reason for this operator is that if there is a single population or intervention criterion, the And/Or
    operators would simplify to the criterion itself. In that case, the _whole_ population or intervention expression of
    the respective population/intervention pair, which should be written to the database with criterion_id = None, would
    be lost (i.e. not written, because there is no graph execution node that would perform this. This operator prevents
    that.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "NonSimplifiableAnd":
        """
        Create a new NonSimplifiableAnd object.
        """
        return super().__new__(cls, *args, **kwargs)


# todo: can this be removed because IntervalWithType is now implemented? I would think so.
class NoDataPreservingAnd(BooleanFunction):
    """
    A And object represents a logical AND operation.

    See Task.handle_no_data_preserving_operator for the rules. Currently, the only difference between this operator
    and the And operator is that during handling of this operator, the negative intervals are added explicitly.
    """


class NoDataPreservingOr(BooleanFunction):
    """
    A Or object represents a logical OR operation.

    See Task.handle_no_data_preserving_operator for the rules. Currently, the only difference between this operator
    and the And operator is that during handling of this operator, the negative intervals are added explicitly.
    """


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


class And(BooleanFunction, sympy.And):
    """
    A And object represents a logical AND operation.
    Extended to include a category attribute.
    """


class Or(BooleanFunction, sympy.Or):
    """
    A Or object represents a logical OR operation.
    Extended to include a category attribute.
    """


class Not(BooleanFunction, sympy.Not):
    """
    A Not object represents a logical NOT operation.
    Extended to include a category attribute.
    """


class Symbol(sympy.Symbol):
    """
    A Symbol object represents a symbol in a symbolic expression.
    Extended to include a criterion and a category attribute.

    :param name: The name of the symbol.
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
