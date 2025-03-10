from abc import ABC, abstractmethod
from datetime import time
from typing import Any, Callable, Dict, Type, cast

from execution_engine.omop.criterion.combination.temporal import TimeIntervalType
from execution_engine.omop.serializable import Serializable

_EXPR_CLASSES: dict[str, Type["BaseExpr"]] = {}


def register_expr_class(cls: Type["BaseExpr"]) -> Type["BaseExpr"]:
    """Decorator to register expression classes by their class name."""
    _EXPR_CLASSES[cls.__name__] = cls
    return cls


class BaseExpr(ABC):
    """
    Base class for expressions and symbols, defining common properties.
    """

    args: tuple

    @classmethod
    def _recreate(cls, args: Any, kwargs: dict) -> "Expr":
        """
        Recreate an expression from its arguments.
        """
        return cast(Expr, cls(*args, **kwargs))

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

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return self._recreate, (self.args, self.get_instance_variables())

    def get_instance_variables(self, immutable: bool = False) -> dict | tuple:
        """
        Return all instance variables of the object.

        If immutable is True, return as an immutable tuple of key-value pairs.
        If immutable is False, return as a mutable dictionary.
        """
        instance_vars = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_")  # Exclude private or special attributes
            and key != "args"
        }

        if immutable:
            return tuple(sorted(instance_vars.items()))
        else:
            return instance_vars

    def dict(self) -> dict:
        """
        Serialize this expression (and its children) into a dict.
        """
        data: dict[str, Any] = {"type": self.__class__.__name__}
        # Store the class name, so we know which subclass to instantiate on from_dict

        # Store instance variables (like thresholds, category, etc.) except for args
        # (They come from get_instance_variables in your snippet)
        instance_vars = self.get_instance_variables(immutable=False)
        for key in instance_vars:
            if isinstance(instance_vars[key], Serializable):
                data[key] = instance_vars[key].dict()
            else:
                data[key] = instance_vars[key]

        # Also store the expression's children by recursing
        if self.args:
            serialized_args = []
            for arg in self.args:
                if isinstance(arg, (BaseExpr, Serializable)):
                    serialized_args.append(arg.dict())  # Recursively serialize
                else:
                    # If non-expression objects appear in .args, store them directly
                    serialized_args.append(arg)
            data["args"] = serialized_args

        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseExpr":
        """
        Rebuild an expression (of the correct subclass) from the given dict.
        """
        expr_type = data["type"]

        # Find the actual subclass
        if expr_type not in _EXPR_CLASSES:
            raise ValueError(f"No registered expression class named '{expr_type}'")
        sub_cls = _EXPR_CLASSES[expr_type]

        # Deserialize child expressions
        children = []
        for arg_data in data.get("args", []):
            if isinstance(arg_data, dict) and "type" in arg_data:
                # It's a nested BaseExpr
                child_expr = cls.from_dict(arg_data)
                children.append(child_expr)
            else:
                # It's a literal (int, str, etc.)
                children.append(arg_data)

        instance_vars = {
            key: val for key, val in data.items() if key not in ("type", "args")
        }

        return sub_cls._recreate(tuple(children), instance_vars)


class Expr(BaseExpr):
    """
    Class for expressions that require a category.
    """

    # todo: isn't this now a bit redundant with the BaseExpr class? (because we've removed category)

    def __new__(cls, *args: Any) -> "Expr":
        """
        Initialize an expression with given arguments.

        :param args: Arguments for the expression.
        """
        self = cast(Expr, super().__new__(cls))

        # we must not allow the __init__ function because of possible infinite recursion when using the __new__ function
        # (see https://pdarragh.github.io/blog/2017/05/22/oddities-in-pythons-new-method/)
        if "__init__" in cls.__dict__:
            raise AttributeError(
                f"__init__ is not allowed in subclass {cls.__name__} of BaseExpr"
            )

        self.args = args

        return self

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}({', '.join(map(repr, self.args))})"

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
        return hash(
            (self.__class__, self.args, self.get_instance_variables(immutable=True))
        )

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


@register_expr_class
class Symbol(BaseExpr):
    """
    Class representing a symbolic variable.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Symbol":
        """
        Initialize a symbol.
        """
        self = cast(Symbol, super().__new__(cls))
        self.args = ()

        return self

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


@register_expr_class
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
        return (
            isinstance(other, self.__class__)
            and self.args == other.args
            and self.get_instance_variables(immutable=True)
            == other.get_instance_variables(immutable=True)
        )

    # Needs to be defined again (although it is the same as in Expr) because we define __eq__ here
    def __hash__(self) -> int:
        """
        Get the hash of this operator.

        :return: Hash of the operator.
        """
        return super().__hash__()

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
            return "(" + f" {self._repr_join_str} ".join(map(repr, self.args)) + ")"
        else:
            return super().__repr__()


@register_expr_class
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

        return super().__new__(cls, *args, **kwargs)


@register_expr_class
class And(BooleanFunction):
    """
    Class representing a logical AND operation.
    """

    _repr_join_str = "&"

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseExpr:
        """
        Create a new And object.
        """
        if len(args) == 1 and isinstance(args[0], BaseExpr):
            return args[0]

        return super().__new__(cls, *args, **kwargs)


@register_expr_class
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

        return cast(Not, super().__new__(cls, *args, **kwargs))


@register_expr_class
class Count(BooleanFunction, ABC):
    """
    Class representing a logical COUNT operation.

    Adds a "threshold" parameter of type int.

    This class should not be instantiated directly, but rather through one of its subclasses.
    """

    count_min: int | None = None
    count_max: int | None = None


@register_expr_class
class MinCount(Count):
    """
    Class representing a logical MIN_COUNT operation.
    """

    def __new__(cls, *args: Any, threshold: int | None, **kwargs: Any) -> "MinCount":
        """
        Create a new MinCount object.
        """
        self = cast(MinCount, super().__new__(cls, *args, **kwargs))
        self.count_min = threshold
        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (self.args, {"threshold": self.count_min}),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


@register_expr_class
class MaxCount(Count):
    """
    Class representing a logical MAX_COUNT operation.
    """

    def __new__(cls, *args: Any, threshold: int | None, **kwargs: Any) -> "MaxCount":
        """
        Create a new MaxCount object.
        """
        self = cast(MaxCount, super().__new__(cls, *args, **kwargs))
        self.count_max = threshold
        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class and arguments.
        """
        return (
            self._recreate,
            (self.args, {"threshold": self.count_max}),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_max}; {', '.join(map(repr, self.args))})"


@register_expr_class
class ExactCount(Count):
    """
    Class representing a logical EXACT_COUNT operation.
    """

    def __new__(cls, *args: Any, threshold: int | None, **kwargs: Any) -> "ExactCount":
        """
        Create a new ExactCount object.
        """
        self = cast(ExactCount, super().__new__(cls, *args, **kwargs))
        self.count_min = threshold
        self.count_max = threshold
        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (self.args, {"threshold": self.count_min}),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


@register_expr_class
class CappedCount(BooleanFunction, ABC):
    """
    Base class representing a COUNT operation with an upper cap.

    This class distinguishes COUNT operations that are subject to an implicit
    maximum constraint, ensuring that they do not exceed what is achievable
    given external limitations.

    Unlike regular COUNT operations, the threshold in this class is not assumed
    to be unbounded. However, no explicit handling of the maximum occurs here;
    it is enforced externally.

    This class should not be instantiated directly but used as a base for specific
    capped count operations like CappedMinCount.
    """

    count_min: int | None = None
    count_max: int | None = None


@register_expr_class
class CappedMinCount(CappedCount):
    """
    Class representing a MIN_COUNT operation with an implicit upper cap.

    This behaves like MinCount but acknowledges that the minimum required count
    is subject to an external upper constraint. If the requested threshold exceeds
    what is achievable, the actual threshold will be limited to the maximum possible
    count, which is determined externally.

    The enforcement of this cap does not occur within this class; rather, it is
    expected to be handled by the surrounding logic.

    The threshold parameter defines the minimum number of overlapping intervals
    required, but in practice, it will not exceed the externally imposed cap.
    """

    def __new__(
        cls, *args: Any, threshold: int | None, **kwargs: Any
    ) -> "CappedMinCount":
        """
        Create a new CappedMinCount object.
        """
        self = cast(CappedMinCount, super().__new__(cls, *args, **kwargs))
        self.count_min = threshold
        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g., when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (self.args, {"threshold": self.count_min}),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


@register_expr_class
class AllOrNone(BooleanFunction):
    """
    Class representing a logical ALL_OR_NONE operation.
    """


@register_expr_class
class TemporalCount(BooleanFunction, ABC):
    """
    Class representing a logical COUNT operation.

    Adds a "threshold" parameter of type int.

    This class should not be instantiated directly, but rather through one of its subclasses.
    """

    count_min: int | None = None
    count_max: int | None = None
    start_time: time | None = None
    end_time: time | None = None
    interval_type: TimeIntervalType | None = None
    interval_criterion: BaseExpr | None = None


@register_expr_class
class TemporalMinCount(TemporalCount):
    """
    Class representing a logical temporal MIN_COUNT operation.
    """

    def __new__(
        cls,
        *args: Any,
        threshold: int | None,
        start_time: time | None,
        end_time: time | None,
        interval_type: TimeIntervalType | None,
        interval_criterion: BaseExpr | None,
        **kwargs: Any,
    ) -> "TemporalMinCount":
        """
        Create a new MinCount object.
        """
        self = cast(TemporalMinCount, super().__new__(cls, *args, **kwargs))
        self.count_min = threshold
        self.start_time = start_time
        self.end_time = end_time
        self.interval_type = interval_type
        self.interval_criterion = interval_criterion

        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (
                self.args,
                {
                    "threshold": self.count_min,
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "interval_type": self.interval_type,
                    "interval_criterion": self.interval_criterion,
                },
            ),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """

        if self.start_time is not None and self.end_time is not None:
            interval = f"{self.start_time} - {self.end_time}"
        elif self.interval_type is not None:
            interval = self.interval_type.name
        elif self.interval_criterion is not None:
            interval = repr(self.interval_criterion)
        else:
            interval = "None"

        return f"{self.__class__.__name__}(interval={interval}; threshold={self.count_min}; {', '.join(map(repr, self.args))})"


@register_expr_class
class TemporalMaxCount(TemporalCount):
    """
    Class representing a logical MAX_COUNT operation.
    """

    def __new__(
        cls,
        *args: Any,
        threshold: int | None,
        start_time: time | None,
        end_time: time | None,
        interval_type: TimeIntervalType | None,
        interval_criterion: BaseExpr | None,
        **kwargs: Any,
    ) -> "TemporalMaxCount":
        """
        Create a new MaxCount object.
        """
        self = cast(TemporalMaxCount, super().__new__(cls, *args, **kwargs))
        self.count_max = threshold
        self.start_time = start_time
        self.end_time = end_time
        self.interval_type = interval_type
        self.interval_criterion = interval_criterion

        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (
                self.args,
                {
                    "threshold": self.count_max,
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "interval_type": self.interval_type,
                    "interval_criterion": self.interval_criterion,
                },
            ),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """

        if self.start_time is not None and self.end_time is not None:
            interval = f"{self.start_time} - {self.end_time}"
        elif self.interval_type is not None:
            interval = self.interval_type.name
        elif self.interval_criterion is not None:
            interval = repr(self.interval_criterion)
        else:
            interval = "None"

        return f"{self.__class__.__name__}(interval={interval}; threshold={self.count_max}; {', '.join(map(repr, self.args))})"


@register_expr_class
class TemporalExactCount(TemporalCount):
    """
    Class representing a logical EXACT_COUNT operation.
    """

    def __new__(
        cls,
        *args: Any,
        threshold: int | None,
        start_time: time | None,
        end_time: time | None,
        interval_type: TimeIntervalType | None,
        interval_criterion: BaseExpr | None,
        **kwargs: Any,
    ) -> "TemporalExactCount":
        """
        Create a new ExactCount object.
        """
        self = cast(TemporalExactCount, super().__new__(cls, *args, **kwargs))
        self.count_min = threshold
        self.count_max = threshold
        self.start_time = start_time
        self.end_time = end_time
        self.interval_type = interval_type
        self.interval_criterion = interval_criterion

        return self

    def __reduce__(self) -> tuple[Callable, tuple]:
        """
        Reduce the expression to its arguments.

        Required for pickling (e.g. when using multiprocessing).

        :return: Tuple of the class, and arguments.
        """
        return (
            self._recreate,
            (
                self.args,
                {
                    "threshold": self.count_min,
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "interval_type": self.interval_type,
                    "interval_criterion": self.interval_criterion,
                },
            ),
        )

    def __repr__(self) -> str:
        """
        Represent the expression in a readable format.
        """

        if self.start_time is not None and self.end_time is not None:
            interval = f"{self.start_time} - {self.end_time}"
        elif self.interval_type is not None:
            interval = self.interval_type.name
        elif self.interval_criterion is not None:
            interval = repr(self.interval_criterion)
        else:
            interval = "None"

        return f"{self.__class__.__name__}(interval={interval}; threshold={self.count_min}; {', '.join(map(repr, self.args))})"


@register_expr_class
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
        return cast(NonSimplifiableAnd, super().__new__(cls, *args, **kwargs))


# todo: can we rename to more meaningful name?
@register_expr_class
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
        return cast(NoDataPreservingAnd, super().__new__(cls, *args, **kwargs))


@register_expr_class
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
        return cast(NoDataPreservingOr, super().__new__(cls, *args, **kwargs))


@register_expr_class
class LeftDependentToggle(BooleanFunction):
    """
    A LeftDependentToggle object represents a logical AND operation if the left operand is positive,
    otherwise it returns NOT_APPLICABLE.
    """

    def __new__(
        cls, left: BaseExpr, right: BaseExpr, **kwargs: Any
    ) -> "LeftDependentToggle":
        """
        Initialize a LeftDependentToggle object.
        """
        return cast(LeftDependentToggle, super().__new__(cls, left, right, **kwargs))

    @property
    def left(self) -> Expr:
        """Returns the left operand"""
        return self.args[0]

    @property
    def right(self) -> Expr:
        """Returns the right operand"""
        return self.args[1]


@register_expr_class
class ConditionalFilter(BooleanFunction):
    """
    A ConditionalFilter object returns the right operand if the left operand is POSITIVE,
    and NEGATIVE otherwise


    A conditional filter returns `right` iff `left` is POSITIVE, otherwise NEGATIVE.

    | left     | right    | Result   |
    |----------|----------|----------|
    | NEGATIVE |    *     | NEGATIVE |
    | NO_DATA  |    *     | NEGATIVE |
    | POSITIVE | POSITIVE | POSITIVE |
    | POSITIVE | NEGATIVE | NEGATIVE |
    | POSITIVE | NO_DATA  | NO_DATA  |
    """

    def __new__(
        cls, left: BaseExpr, right: BaseExpr, **kwargs: Any
    ) -> "ConditionalFilter":
        """
        Initialize a ConditionalFilter object.
        """
        return cast(ConditionalFilter, super().__new__(cls, left, right, **kwargs))

    @property
    def left(self) -> Expr:
        """Returns the left operand"""
        return self.args[0]

    @property
    def right(self) -> Expr:
        """Returns the right operand"""
        return self.args[1]
