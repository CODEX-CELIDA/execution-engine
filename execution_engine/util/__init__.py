import datetime
from abc import ABCMeta
from typing import Any


def datetime_converter(obj: Any) -> Any:
    """
    Convert datetime objects to ISO format strings.

    Used in json.dumps() to serialize datetime objects.
    """
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    return obj


class AbstractPrivateMethods(ABCMeta):
    """
    A metaclass that prevents overriding of methods decorated with @typing.final.
    """

    def __new__(mcs, name: str, bases: tuple, class_dict: dict) -> Any:
        """
        Instantiate a new class.

        Checks for __final__ attribute set on methods of parent classes (via @typing.final decorator)
        and raises an error if a child class tries to override them.
        """
        private = {
            key: base.__qualname__
            for base in bases
            for key, value in vars(base).items()
            if callable(value) and getattr(value, "__final__", False)
        }

        if any(key in private for key in class_dict):
            message = ", ".join([f"{v}.{k}" for k, v in private.items()])
            raise RuntimeError(f"Methods {message} may not be overriden")
        return super().__new__(mcs, name, bases, class_dict)
