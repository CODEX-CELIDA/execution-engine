from abc import ABC, abstractmethod
from typing import Any

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion


class BaseExpr(ABC):
    """
    Base class for expressions and symbols, defining common properties.
    """

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """
        Check if this expression is equal to another expression.
        """
        raise NotImplementedError("__eq__ must be implemented by subclasses")

    @abstractmethod
    def __hash__(self) -> int:
        """
        Get the hash of this expression.
        """
        raise NotImplementedError("__hash__ must be implemented by subclasses")

    @property
    def is_Atom(self) -> bool:
        """
        Check if the object is an atom (not divisible into smaller parts).

        :return: True if atom, False otherwise. To be overridden in subclasses.
        """
        raise NotImplementedError("is_Atom must be implemented by subclasses")

    @property
    def is_Not(self) -> bool:
        """
        Check if the object is a Not type.

        :return: True if Not type, False otherwise. To be overridden in subclasses.
        """
        raise NotImplementedError("is_Not must be implemented by subclasses")


class Expr(BaseExpr):
    """
    Class for expressions that require a category.
    """

    def __init__(self, *args: Any, category: CohortCategory):
        """
        Initialize an expression with given arguments and a mandatory category.

        :param args: Arguments for the expression.
        :param category: Mandatory category of the expression.
        """
        self.args = args
        self.category = category

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}({', '.join(map(repr, self.args))}, category='{self.category}')"

    def __eq__(self, other: Any) -> bool:
        """
        Check if this expression is equal to another expression.

        :param other: The other expression.
        :return: True if equal, False otherwise.
        """
        return isinstance(other, self.__class__) and hash(self) == hash(other)

    def __hash__(self) -> int:
        """
        Get the hash of this expression.

        :return: Hash of the expression.
        """
        return hash((self.__class__, self.args, self.category))

    @property
    def is_Atom(self) -> bool:
        """
        Check if the expression is an atom. Returns False for general expressions.

        :return: False for Expr.
        """
        return False

    @property
    def is_Not(self) -> bool:
        """
        Check if the expression is a Not type.

        :return: True if Not type, False otherwise.
        """
        return isinstance(self, Not)


class Symbol(BaseExpr):
    """
    Class representing a symbolic variable.
    """

    criterion: Criterion

    def __init__(self, criterion: Criterion) -> None:
        """
        Initialize a symbol.

        :param criterion: The criterion of the symbol.
        """
        self.args = ()
        self.criterion = criterion

    def __eq__(self, other: Any) -> bool:
        """
        Check if this symbol is equal to another symbol.

        :param other: The other symbol.
        :return: True if equal, False otherwise.
        """
        return isinstance(other, Symbol) and self.criterion == other.criterion

    def __hash__(self) -> int:
        """
        Get the hash of this symbol.

        :return: Hash of the symbol.
        """
        return hash(self.criterion)

    @property
    def category(self) -> CohortCategory:
        """
        Get the cohort category of the symbol.

        :return: The cohort category.
        """
        return self.criterion.category

    def __repr__(self) -> str:
        """
        Represent the symbol.

        :return: Name of the symbol.
        """
        return self.criterion.description()

    @property
    def is_Atom(self) -> bool:
        """
        Check if the Symbol is an atom. Always returns True for Symbol.

        :return: True as Symbol is always an atom.
        """
        return True

    @property
    def is_Not(self) -> bool:
        """
        Check if the Symbol is a Not type. Always returns False for Symbol.

        :return: False as Symbol is never a Not type.
        """
        return False


class BooleanFunction(Expr):
    """
    Base class for boolean functions like OR, AND, and NOT.
    """

    _repr_join_str: str | None = None

    def __eq__(self, other: Any) -> bool:
        """
        Check if this operator is equal to another operator.

        :param other: The other operator.
        :return: True if equal, False otherwise.
        """
        return isinstance(other, self.__class__) and self.args == other.args

    def __hash__(self) -> int:
        """
        Get the hash of this operator.

        :return: Hash of the operator.
        """
        return hash((self.__class__, self.args))

    @property
    def is_Atom(self) -> bool:
        """
        Boolean functions are not atoms.

        :return: False
        """
        return False

    @property
    def is_Not(self) -> bool:
        """
        Check if the BooleanFunction is a Not type.

        :return: True if Not type, False otherwise.
        """
        return isinstance(self, Not)

    def __repr__(self) -> str:
        """
        Represent the BooleanFunction in a readable format.
        """
        if self._repr_join_str is not None:
            return f" {self._repr_join_str} ".join(map(repr, self.args))
        else:
            return super().__repr__()


class Or(BooleanFunction):
    """
    Class representing a logical OR operation.
    """

    _repr_join_str = "|"

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseExpr:
        """
        Create a new Or object.
        """
        if len(args) == 1 and isinstance(args[0], BaseExpr):
            return args[0]

        return super().__new__(cls)


class And(BooleanFunction):
    """
    Class representing a logical AND operation.
    """

    _repr_join_str = "&"

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseExpr:
        """
        Create a new Or object.
        """
        if len(args) == 1 and isinstance(args[0], BaseExpr):
            return args[0]
        return super().__new__(cls)


class Not(BooleanFunction):
    """
    Class representing a logical NOT operation.
    """

    def __repr__(self) -> str:
        """
        Represent the NOT operation as a string.
        """
        return f"~{self.args[0]}"

    def __new__(cls, *args: Any, **kwargs: Any) -> "Not":
        """
        Create a new Or object.
        """
        if len(args) > 1:
            raise ValueError("Not can only have one argument")

        return super().__new__(cls)


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

    def __eq__(self, other: Any) -> bool:
        """
        Check if this operator is equal to another operator.

        This always yields false to prevent combination of two NonSimplifiableAnd operators when merging a graph.
        """
        return False

    def __hash__(self) -> int:
        """
        Get the hash of this operator.

        Required because __eq__ yields always False -- and we need distinct hashes for distinct objects, as this
        operator should not be merged.
        """
        return id(self)

    def __new__(cls, *args: Any, **kwargs: Any) -> "NonSimplifiableAnd":
        """
        Create a new NonSimplifiableAnd object.
        """
        return super().__new__(cls)


# todo: can this be removed because IntervalWithType is now implemented? I would think so.
class NoDataPreservingAnd(BooleanFunction):
    """
    A And object represents a logical AND operation.

    See Task.handle_no_data_preserving_operator for the rules. Currently, the only difference between this operator
    and the And operator is that during handling of this operator, the negative intervals are added explicitly.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "NoDataPreservingAnd":
        """
        Create a new NoDataPreservingAnd object.
        """
        return super().__new__(cls)


class NoDataPreservingOr(BooleanFunction):
    """
    A Or object represents a logical OR operation.

    See Task.handle_no_data_preserving_operator for the rules. Currently, the only difference between this operator
    and the And operator is that during handling of this operator, the negative intervals are added explicitly.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "NoDataPreservingOr":
        """
        Create a new NoDataPreservingOr object.
        """
        return super().__new__(cls)


class LeftDependentToggle(BooleanFunction):
    """
    A LeftDependentToggle object represents a logical AND operation if the left operand is positive,
    otherwise it returns NOT_APPLICABLE.
    """

    def __init__(self, left: BaseExpr, right: BaseExpr, **kwargs: Any) -> None:
        """
        Initialize a LeftDependentToggle object.
        """
        super().__init__(left, right, **kwargs)

    @property
    def left(self) -> Expr:
        """Returns the left operand"""
        return self.args[0]

    @property
    def right(self) -> Expr:
        """Returns the right operand"""
        return self.args[1]
