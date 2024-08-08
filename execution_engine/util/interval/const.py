"""
Definition copied from portion package: https://pypi.org/project/portion/ (c) AlexandreDecan
"""

from enum import Enum
from typing import Any


class Bound(Enum):
    """
    Bound types, either CLOSED for inclusive, or OPEN for exclusive.
    """

    CLOSED = True
    OPEN = False

    def __bool__(self) -> bool:
        """
        Return the boolean value of the bound.
        """
        raise ValueError("The truth value of a bound is ambiguous.")

    def __invert__(self) -> "Bound":
        """
        Return the opposite bound.
        """
        return Bound.CLOSED if self is Bound.OPEN else Bound.OPEN

    def __str__(self) -> str:
        """
        Return the string representation of the bound.
        """
        return self.name

    def __repr__(self) -> str:
        """
        Return the string representation of the bound.
        """
        return self.name


class _Singleton:
    __instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "_Singleton":
        if not cls.__instance:
            cls.__instance = super(_Singleton, cls).__new__(cls)
        return cls.__instance


class _PInf(_Singleton):
    """
    Represent positive infinity.
    """

    def __neg__(self) -> "_NInf":
        """
        Return negative infinity.
        """
        return _NInf()

    def __lt__(self, o: Any) -> bool:
        """
        Positive infinity is always greater than any other object.
        """
        return False

    def __le__(self, o: Any) -> bool:
        """
        Positive infinity is always greater than any other object, and equal to itself.
        """
        return isinstance(o, _PInf)

    def __gt__(self, o: Any) -> bool:
        """
        Positive infinity is greater than any other object but positive infinity.
        """
        return not isinstance(o, _PInf)

    def __ge__(self, o: Any) -> bool:
        """
        Return True if the object is positive infinity, False
        """
        return True

    def __eq__(self, o: Any) -> bool:
        """
        Return True if the object is positive infinity, False
        """
        return isinstance(o, _PInf)

    def __repr__(self) -> str:
        """
        Return the string representation
        """
        return "+inf"

    def __hash__(self) -> int:
        """
        Return the hash value.
        """
        return hash(float("+inf"))


class _NInf(_Singleton):
    """
    Represent negative infinity.
    """

    def __neg__(self) -> "_PInf":
        """
        Return positive infinity.
        """
        return _PInf()

    def __lt__(self, o: Any) -> bool:
        """
        Return True if the object is negative infinity, False
        """
        return not isinstance(o, _NInf)

    def __le__(self, o: Any) -> bool:
        """
        Return True if the object is negative infinity, False
        """
        return True

    def __gt__(self, o: Any) -> bool:
        """
        Return True if the object is negative infinity, False
        """
        return False

    def __ge__(self, o: Any) -> bool:
        """
        Return True if the object is negative infinity, False
        """
        return isinstance(o, _NInf)

    def __eq__(self, o: Any) -> bool:
        """
        Return True if the object is negative infinity, False
        """
        return isinstance(o, _NInf)

    def __repr__(self) -> str:
        """
        Return the string representation.
        """
        return "-inf"

    def __hash__(self) -> int:
        """
        Return the hash value.
        """
        return hash(float("-inf"))


inf = _PInf()
