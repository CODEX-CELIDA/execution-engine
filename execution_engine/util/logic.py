from abc import abstractmethod
from datetime import time
from typing import Any, Dict, Iterator, Self, cast

from execution_engine.util.enum import TimeIntervalType
from execution_engine.util.serializable import Serializable, SerializableABC


def arg_to_dict(arg: Any, include_id: bool) -> dict:
    """
    Convert an argument to a dictionary representation.

    :param arg: The argument to convert.
    :param include_id: Whether to include the ID in the dictionary.
    :return: Dictionary representation of the argument.
    """
    return arg.dict(include_id=include_id) if isinstance(arg, BaseExpr) else arg


class BaseExpr(Serializable):
    """
    Base class for expressions and symbols, defining common properties.
    """

    args: tuple

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
    Class for expressions that are not Symbols
    """

    # todo: isn't this now a bit redundant with the BaseExpr class? (because we've removed category)

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

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute on the object.

        This is overridden to prevent setting attributes on the object.
        """
        if name in self.__dict__ and name not in ["args", "_init_args"]:
            raise AttributeError(
                f"Cannot update attributes on {self.__class__.__name__}"
            )
        super().__setattr__(name, value)

    @abstractmethod
    def update_args(self, *args: Any) -> None:
        """
        Update the arguments of the expression.

        :param args: The new arguments.
        """
        raise NotImplementedError("update_args must be implemented by subclasses")

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

    def __str__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"

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

    def atoms(self) -> Iterator["Symbol"]:
        """
        Get all symbols in the expression.
        """

        def traverse(expr: BaseExpr) -> Iterator[Symbol]:
            if expr.is_Atom:
                assert isinstance(expr, Symbol), f"Expected Symbol, got {expr}"
                yield expr

            for sub_expr in expr.args:
                yield from traverse(sub_expr)

        yield from traverse(self)

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "data": {
                "args": [arg_to_dict(arg, include_id=include_id) for arg in self.args],
            },
        }

        if include_id and self._id is not None:
            data["data"]["_id"] = self._id

        return data


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

    def get_instance_variables(self, immutable: bool = False) -> Dict[str, Any] | tuple:
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


class BooleanFunction(Expr):
    """
    Base class for boolean functions like OR, AND, and NOT.
    """

    _repr_join_str: str | None = None

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


class UnaryOperator(BooleanFunction):
    """
    Base class for unary operators.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "UnaryOperator":
        """
        Create a new UnaryOperator object.
        """
        if len(args) > 1:
            raise ValueError(f"{cls.__name__} can only have one argument")

        return cast(UnaryOperator, super().__new__(cls, *args, **kwargs))

    def update_args(self, *args: Any) -> None:
        """
        Update the arguments of the expression.

        :param args: The new arguments.
        """
        self.args = args


class CommutativeOperator(BooleanFunction, SerializableABC):
    """
    Base class for commutative operators.
    """

    def update_args(self, *args: Any) -> None:
        """
        Update the arguments of the expression.

        :param args: The new arguments.
        """
        self.args = args

    def __new__(cls, *args: Any, **kwargs: Any) -> "CommutativeOperator":
        """
        Create a new CommutativeOperator object.
        """
        return cast(CommutativeOperator, super().__new__(cls, *args, **kwargs))


class Or(CommutativeOperator):
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


class And(CommutativeOperator):
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


class Not(UnaryOperator):
    """
    Class representing a logical NOT operation.
    """

    def __str__(self) -> str:
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


class Count(CommutativeOperator, SerializableABC):
    """
    Class representing a logical COUNT operation.

    Adds a "threshold" parameter of type int.

    This class should not be instantiated directly, but rather through one of its subclasses.
    """

    count_min: int | None = None
    count_max: int | None = None


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

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """

        data = super().dict(include_id=include_id)
        data["data"].update({"threshold": self.count_min})

        return data

    def __str__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


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

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = super().dict(include_id=include_id)
        data["data"].update({"threshold": self.count_max})
        return data

    def __str__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_max}; {', '.join(map(repr, self.args))})"


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

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = super().dict(include_id=include_id)
        data["data"].update({"threshold": self.count_min})
        return data

    def __str__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


class CappedCount(CommutativeOperator, SerializableABC):
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

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = super().dict(include_id=include_id)
        data["data"].update({"threshold": self.count_min})
        return data

    def __str__(self) -> str:
        """
        Represent the expression in a readable format.
        """
        return f"{self.__class__.__name__}(threshold={self.count_min}; {', '.join(map(repr, self.args))})"


class AllOrNone(CommutativeOperator):
    """
    Class representing a logical ALL_OR_NONE operation.
    """


class TemporalCount(CommutativeOperator, SerializableABC):
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

    def __new__(
        cls,
        *args: Any,
        threshold: int | None,
        start_time: time | None = None,
        end_time: time | None = None,
        interval_type: TimeIntervalType | None = None,
        interval_criterion: BaseExpr | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a new TemporalCount object.
        """
        TemporalCount._validate_time_inputs(
            start_time, end_time, interval_type, interval_criterion
        )

        if interval_criterion:
            # we need to add the interval_criterion to the list of arguments of this criterion in order to have
            # it properly processed
            args += (interval_criterion,)

        self = cast(Self, super().__new__(cls, *args, **kwargs))

        self.count_min = threshold
        self.start_time = (
            time.fromisoformat(start_time)  # type: ignore[arg-type]
            if isinstance(start_time, str)
            else start_time
        )
        self.end_time = (
            time.fromisoformat(end_time)  # type: ignore[arg-type]
            if isinstance(end_time, str)
            else end_time
        )
        self.interval_type = interval_type
        self.interval_criterion = interval_criterion

        return self

    @classmethod
    def _validate_time_inputs(
        self,
        start_time: time | None,
        end_time: time | None,
        interval_type: TimeIntervalType | None,
        interval_criterion: BaseExpr | None,
    ) -> None:

        if interval_type:
            if start_time is not None or end_time is not None:
                raise ValueError(
                    "start_time/end_time cannot be used together with interval_type"
                )
            if interval_criterion is not None:
                raise ValueError(
                    "interval_criterion cannot be used together with interval_type"
                )
            # Validate the interval_type if needed
            self.interval_type = interval_type
            self.start_time = None
            self.end_time = None

        elif start_time or end_time:
            # Must have start_time and end_time
            if start_time is None or end_time is None:
                raise ValueError(
                    "Either interval_type or interval_criterion or both start_time & end_time must be provided"
                )
            if interval_criterion is not None:
                raise ValueError(
                    "interval_criterion cannot be used together with start_time/end_time"
                )
            if start_time >= end_time:
                raise ValueError("start_time must be less than end_time")

        elif interval_criterion and not isinstance(interval_criterion, (BaseExpr)):
            raise ValueError(
                f"Invalid criterion - expected Criterion or CriterionCombination, got {type(interval_criterion)}"
            )

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = super().dict(include_id=include_id)
        data["data"].update(
            {
                "threshold": self.count_min,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "interval_type": self.interval_type,
                "interval_criterion": (
                    self.interval_criterion.dict(include_id=include_id)
                    if self.interval_criterion
                    else None
                ),
            }
        )
        return data

    def __str__(self) -> str:
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


class TemporalMinCount(TemporalCount):
    """
    Class representing a logical temporal MIN_COUNT operation.
    """


class TemporalMaxCount(TemporalCount):
    """
    Class representing a logical MAX_COUNT operation.
    """


class TemporalExactCount(TemporalCount):
    """
    Class representing a logical EXACT_COUNT operation.
    """


class NonSimplifiableAnd(CommutativeOperator):
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
        return cast(NonSimplifiableAnd, super().__new__(cls, *args, **kwargs))


# todo: can we rename to more meaningful name?
class NoDataPreservingAnd(CommutativeOperator):
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


class NoDataPreservingOr(CommutativeOperator):
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


class BinaryNonCommutativeOperator(BooleanFunction, SerializableABC):
    """
    Base class for binary non-commutative operators.

    This class should not be instantiated directly but used as a base for specific
    binary non-commutative operators like LeftDependentToggle.
    """

    def update_args(self, *args: Any) -> None:
        """
        Update the arguments of the expression.

        :param args: The new arguments.
        """
        if len(args) != 2:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly two arguments"
            )
        self.args = args

    def __new__(
        cls, left: BaseExpr, right: BaseExpr, **kwargs: Any
    ) -> "BinaryNonCommutativeOperator":
        """
        Create a new BinaryNonCommutativeOperator object.
        """
        return cast(
            BinaryNonCommutativeOperator, super().__new__(cls, left, right, **kwargs)
        )

    @property
    def left(self) -> Expr:
        """Returns the left operand"""
        return self.args[0]

    @property
    def right(self) -> Expr:
        """Returns the right operand"""
        return self.args[1]

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = super().dict(include_id=include_id)
        del data["data"]["args"]
        data["data"].update(
            {
                "left": arg_to_dict(self.left, include_id=include_id),
                "right": arg_to_dict(self.right, include_id=include_id),
            }
        )
        return data


class LeftDependentToggle(BinaryNonCommutativeOperator):
    """
    A LeftDependentToggle object represents a logical AND operation if the left operand is positive,
    otherwise it returns NOT_APPLICABLE.
    """


class ConditionalFilter(BinaryNonCommutativeOperator):
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
