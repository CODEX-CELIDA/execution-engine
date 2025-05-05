import abc
import inspect
import json
from typing import Any, Dict, Self, final

from pydantic import BaseModel

from execution_engine.util import datetime_converter

__class_registry: dict[str, type] = {}
"""
Registry of classes that can be serialized.
"""


def register_class(cls: type) -> type:
    """
    Register a class for serialization.

    This function may be used as a decorator to register a class in the serialization registry.

    :param cls: The class to register.
    :return: The same class (for decorator chaining).
    :raises ValueError: If the class name is already registered.
    """
    if cls.__name__ in __class_registry:
        raise ValueError(f"Class {cls.__name__} is already registered.")

    __class_registry[cls.__name__] = cls
    return cls


def resolve_class(name: str) -> type:
    """
    Resolve a registered class by name.

    :param name: The name of the class to retrieve.
    :return: The corresponding class object.
    :raises ValueError: If the class is not found in the registry.
    """
    cls = __class_registry.get(name)

    if cls is None:
        raise ValueError(f"Class {name} is not registered.")

    return cls


def is_class_registered(name: str) -> bool:
    """
    Check if a class is registered under the given name.

    :param name: The name of the class to check.
    :return: True if the class is registered, False otherwise.
    """
    return name in __class_registry


class RegisteredPostInitMeta(type):
    """
    Metaclass that automatically registers a class for serialization and calls
    a custom __post_init__ method (if defined) once after regular object initialization.
    """

    def __call__(cls, *args: Any, do_post_init: bool = True, **kwargs: Any) -> Self:
        """
        Create and return a new instance of the class, then call __post_init__ if defined.

        This overrides the default object construction process to perform any
        custom post-initialization logic defined in __post_init__. If __post_init__
        exists, it is called exactly once, after the instance is created.

        :param args: Positional arguments used during object creation.
        :param kwargs: Keyword arguments used during object creation.
        :return: The newly created instance of the class.
        """
        instance = super().__call__(*args, **kwargs)

        if (
            do_post_init
            and hasattr(instance, "__post_init__")
            and not getattr(instance, "_post_initialized", False)
        ):
            instance.__post_init__()
            instance._post_initialized = True

        return instance

    def __new__(mcs, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Self:
        """
        Create and return a new class, registering it in the serialization registry.

        This method intercepts the class creation process itself. It registers the
        newly defined class to __class_registry, allowing instances to be properly
        deserialized later.

        :param name: The name of the newly created class.
        :param bases: A tuple of base classes.
        :param attrs: A dictionary of attributes/methods for the new class.
        :return: The newly created class object.
        :raises ValueError: If the class is already registered.
        """
        new_class = super().__new__(mcs, name, bases, attrs)
        register_class(new_class)
        return new_class


def immutable_setattr(self: Self, key: str, value: Any) -> None:
    """
    Prevent setting attributes on an immutable object.

    This function is assigned to an instance's __setattr__ method in order to enforce
    immutability after object creation. Any attempt to set an attribute on the instance
    after initialization will raise an AttributeError.

    :param self: The instance on which the attribute assignment was attempted.
    :param key: The name of the attribute being set.
    :param value: The value being assigned.
    :raises AttributeError: Always, to enforce immutability.
    """
    raise AttributeError(
        f"Cannot set attribute {key} on immutable object {self.__class__.__name__}"
    )


class Serializable(metaclass=RegisteredPostInitMeta):
    """
    Base class for making objects serializable.

    Stores construction arguments, manages an optional database ID, and provides serialization
    and deserialization to and from dictionaries or JSON.

    Note that Serializable classes are immutable. This means that once an object is created,
    its attributes cannot be changed. This is enforced by overriding the __setattr__ method in
    the __post_init__ method.
    The rationale behind this is to provide a fixed hash value for the object, which is
    calculated only once during the object's lifetime. This is important for caching and
    serialization purposes.
    """

    _id: int | None = None
    """
    The id is used in the database tables.
    """

    _hash: int
    """
    The hash of the object. This is calculated based on the class name and the JSON representation
    of the object. It is used to ensure that the object is immutable.
    """

    def __post_init__(self) -> None:
        """
        Create a new instance of the object.
        """
        self.rehash()

        self.__setattr__ = immutable_setattr  # type: ignore[assignment]

    def set_id(self, value: int, overwrite: bool = False) -> None:
        """
        Assigns the database ID to the object. This can only be done once.

        This ID corresponds to the primary key in the database and is set
        when the object is persisted.

        :param value: The database ID assigned to the object.
        :param overwrite: If True, allows overwriting an existing ID.
        :raises ValueError: If the ID has already been set.
        """
        if self._id is not None and not overwrite:
            raise ValueError("Database ID has already been set!")
        object.__setattr__(self, "_id", value)

    def reset_id(self) -> None:
        """
        Resets the database ID.
        """
        # Circumvents the immutable __setattr__
        object.__setattr__(self, "_id", None)

    @property
    def id(self) -> int:
        """
        Retrieves the database ID of the object.

        This ID is only available after the object has been stored in the database.

        :return: The database ID, or None if the object has not been stored yet.
        """
        if self._id is None:
            raise ValueError("Database ID has not been set yet!")
        return self._id

    def is_persisted(self) -> bool:
        """
        Returns True if the object has been stored in the database.
        """
        return self._id is not None

    def dict(self, include_id: bool = False) -> dict:
        """
        Get a dictionary representation of the object.
        """
        data = {}

        def serialize(val: Any) -> Any:
            if isinstance(val, Serializable):
                return val.dict(include_id=include_id)
            elif isinstance(val, BaseModel):
                return {
                    "type": val.__class__.__name__,
                    "data": val.model_dump(),
                }
            elif isinstance(val, (list, tuple)):
                return type(val)(serialize(item) for item in val)
            return val

        for key, val in self.get_instance_variables().items():  # type: ignore[union-attr]
            data[key] = serialize(val)

        if include_id and self._id is not None:
            data["_id"] = self._id

        return {"type": self.__class__.__name__, "data": data}

    def get_instance_variables(self, immutable: bool = False) -> Dict[str, Any] | tuple:
        """
        Get the instance variables of the object.

        This is only required if the subclass doesn't provide an own implementation of dict().
        """
        raise NotImplementedError(
            "Method get_instance_variables must be implemented in subclasses."
        )

    @final
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an object from a dictionary produced by dict().

        Expected format:
        {
            "type": <class name>,
            "data": {
                "_id": ... ,             (optional)
                "args": [...],           (optional, positional arguments)
                ... any additional keys ...
            }
        }
        """
        class_name = data["type"]
        content = data["data"]

        # Look up the target class from our registry
        sub_cls = resolve_class(class_name)

        # Extract _id if present
        db_id = content.pop("_id", None)

        pos_args = content.pop("args", [])
        var_kwargs = content

        def deserialize_item(item: Any) -> Any:
            if isinstance(item, dict) and "type" in item:
                return cls.from_dict(item)
            return item

        pos_args = [deserialize_item(arg) for arg in pos_args]
        var_kwargs = {k: deserialize_item(v) for k, v in var_kwargs.items()}

        obj = sub_cls(*pos_args, **var_kwargs)

        if db_id is not None:
            obj.set_id(db_id)

        return obj

    def json(self) -> bytes:
        """
        Get a JSON representation of the object.

        The json excludes the id, as this is auto-inserted by the database
        and not known during the creation of the object.
        """
        s_json = self.dict()

        return json.dumps(s_json, default=datetime_converter, sort_keys=True).encode()

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
        """
        Create a combination from a JSON string.
        """
        return cls.from_dict(json.loads(data))

    def __eq__(self, other: Any) -> bool:
        """
        Check if two objects are equal.
        """
        if not isinstance(other, self.__class__):
            return False

        return hash(self) == hash(other)

    def rehash(self) -> None:
        """
        Recalculate the hash of the object.
        """
        self._hash = hash(self.__class__.__name__.encode() + self.json())

    def __hash__(self) -> int:
        """
        Get the hash of the object.
        """
        return self._hash

    # def __reduce__(self) -> Tuple[Callable, tuple]:
    #     """
    #     Support pickling of the object.
    #     """
    #     return self.__class__.from_dict, (self.dict(include_id=True),)

    @final
    def __repr__(self) -> str:
        """
        Get a string representation of the object.
        """
        rep_dict = self.dict(include_id=False)
        data = rep_dict["data"].copy()

        return self.build_repr(rep_dict["type"], data, indent=0)

    def build_repr(self, cls: str, data: Dict, indent: int) -> str:
        """
        Build a string representation of the object.
        """
        indent_unit = "  "

        def pformat_value(val: Any, indent: int) -> str:
            current_indent = indent_unit * indent
            next_indent = indent_unit * (indent + 1)
            # Check for a serialized object (dict with "type" and "data")
            if isinstance(val, dict) and "type" in val and "data" in val:

                if is_class_registered(val["type"]) and issubclass(
                    resolve_class(val["type"]), BaseModel
                ):
                    flat_data = ", ".join(
                        f"{k}={repr(v)}" for k, v in val["data"].items()
                    )
                    return f"{val['type']}({flat_data})"

                return self.build_repr(val["type"], val["data"], indent + 1)
            elif isinstance(val, list):
                if not val:
                    return "[]"
                items = [pformat_value(item, indent + 1) for item in val]
                return (
                    "[\n"
                    + ",\n".join(next_indent + item for item in items)
                    + "\n"
                    + current_indent
                    + "]"
                )
            elif isinstance(val, tuple):
                if not val:
                    return "()"
                # Special case for single-element tuples
                if len(val) == 1:
                    item = pformat_value(val[0], indent + 1)
                    return "(\n" + next_indent + item + ",\n" + current_indent + ")"
                else:
                    items = [pformat_value(item, indent + 1) for item in val]
                    return (
                        "(\n"
                        + ",\n".join(next_indent + item for item in items)
                        + "\n"
                        + current_indent
                        + ")"
                    )
            elif isinstance(val, dict):
                if not val:
                    return "{}"
                items = []
                for k, v in val.items():
                    formatted_v = pformat_value(v, indent + 1)
                    items.append(f"{next_indent}{repr(k)}: {formatted_v}")
                return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"
            else:
                return repr(val)

        # Extract positional arguments (if any)
        pos_args = data.pop("args", [])
        pos_args_str = [pformat_value(arg, indent=indent + 1) for arg in pos_args]

        # Format remaining keyword arguments
        kw_args_str = [
            f"{key}={pformat_value(value, indent=indent+1)}"
            for key, value in data.items()
        ]

        # Combine both sets of arguments
        all_args = pos_args_str + kw_args_str
        outer_indent = indent_unit * indent
        inner_indent = indent_unit * (indent + 1)

        if all_args:
            args_joined = ",\n".join(inner_indent + arg for arg in all_args)
            return f"{cls}(\n{args_joined}\n{outer_indent})"
        else:
            return f"{cls}()"

    def __str__(self) -> str:
        """
        Get a string representation of the object.
        """
        data = {}

        for key, val in self.get_instance_variables().items():  # type: ignore[union-attr]
            data[key] = str(val)

        return f"{self.__class__.__name__}({data})"


def get_constructor_vars(cls: type) -> set[str]:
    """
    Get the variables that are passed to the constructor of a class.

    :param cls: The class.
    :return: A set of variable names.
    """
    self_var = "self"
    original_init = cls.__init__
    if cls.__init__ == object.__init__ and cls.__new__ != object.__new__:
        original_init = cls.__new__
        self_var = "cls"

    sig = inspect.signature(original_init)

    return {arg for arg in sig.parameters if arg != self_var}


class SerializableDataClassMeta(RegisteredPostInitMeta):
    """
    Base class for making objects serializable. Stores construction
    arguments, manages an optional database ID, and provides serialization
    and deserialization to and from dictionaries or JSON.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Creates and returns a new instance of the class, ensuring that arguments
        are assigned to protected attributes. Uses the function signature to bind
        positional and keyword arguments and raises a ValueError if protected
        attributes are missing.

        :param args: Positional arguments for object creation.
        :param kwargs: Keyword arguments for object creation.
        :return: Newly created instance of the class.
        """
        instance = super().__call__(*args, do_post_init=False, **kwargs)

        self_var = "self"
        original_init = cls.__init__
        if cls.__init__ == object.__init__ and cls.__new__ != object.__new__:
            original_init = cls.__new__
            self_var = "cls"

        sig = inspect.signature(original_init)
        if self_var == "cls":
            # For __new__, the first argument should be the class (instance's class)
            bound = sig.bind(cls, *args, **kwargs)
        else:
            # For __init__, the first argument is the instance
            bound = sig.bind(instance, *args, **kwargs)

        bound.arguments.pop(self_var, None)  # Remove 'self' or 'cls'

        protected_instance_vars = set(
            arg
            for arg in vars(instance)
            if arg.startswith("_") and not arg.startswith("__")
        )

        # Check if the set of arguments passed to __init__ is a subset of the protected instance variables
        if not set(f"_{arg}" for arg in bound.arguments) <= protected_instance_vars:
            raise AttributeError(
                "All arguments passed to __init__ must be assigned to the instance as protected attributes"
                " in a SerializableDataClass."
            )

        if hasattr(instance, "__post_init__") and not getattr(
            instance, "_post_initialized", False
        ):
            instance.__post_init__()
            instance._post_initialized = True  # type: ignore[attr-defined]

        return instance


class SerializableABCMeta(RegisteredPostInitMeta, abc.ABCMeta):
    """
    Metaclass that combines SerializableDataClassMeta logic with ABCMeta
    to allow abstract methods in serializable classes.
    """


class SerializableABC(metaclass=SerializableABCMeta):
    """
    Abstract base class for serializable objects. This class allows
    the use of abstract methods in serializable classes.
    """


class SerializableDataClassABCMeta(SerializableDataClassMeta, abc.ABCMeta):
    """
    Metaclass that combines SerializableDataClassMeta logic with ABCMeta
    to allow abstract methods in serializable data classes.
    """


class SerializableDataClass(Serializable, metaclass=SerializableDataClassMeta):
    """
    Serializable data class that ensures arguments passed to __init__
    are protected attributes. Automatically registers subclasses and
    supports dict() and JSON exports.
    """

    def get_instance_variables(self, immutable: bool = False) -> dict | tuple:
        """
        Get a dictionary representation of the criterion.
        """

        if immutable:
            raise NotImplementedError(
                "get_instance_variables() must be implemented in subclasses."
            )

        return {
            var: getattr(self, f"_{var}")
            for var in get_constructor_vars(self.__class__)
        }


class SerializableDataClassABC(
    SerializableDataClass, metaclass=SerializableDataClassABCMeta
):
    """
    Abstract variant of a serializable data class. Requires subclasses
    to implement abstract methods, while keeping the serialization
    features and protected attribute checks.
    """
