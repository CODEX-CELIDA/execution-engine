import abc
import inspect
from typing import Any, Type, TypeVar

T = TypeVar("T", bound="SignatureReprMeta")


class SignatureReprMeta(type):
    """
    A metaclass that automatically captures constructor arguments and generates a __repr__ method.

    This metaclass wraps the `__init__` method of a class to store the arguments passed at instantiation,
    allowing `__repr__` to dynamically generate an informative string representation, omitting default values.

    Features:
    - Captures constructor arguments at instantiation time (`self._init_args`).
    - Automatically generates a `__repr__` if the class does not define one.
    - Ensures that `__repr__` only displays arguments that differ from the default values.
    - Prevents redundant re-capturing when a parent class’s `__init__` is invoked via `super()`.
    - Retains the original `__init__` function signature for accurate introspection.

    Usage:
    ```python
    class MyClass(metaclass=SignatureReprMeta):
        def __init__(self, x, y=10, z=None):
            self.x = x
            self.y = y
            self.z = z

    obj = MyClass(5, z="test")
    print(obj)  # Output: MyClass(x=5, z='test')  (y is omitted since it uses the default 10)
    ```

    """

    def __new__(
        mcs: Type[T], name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> T:
        """
        Wrap the __init__ method and attach a default __repr__ if not defined.
        """
        original_init = namespace.get("__init__")

        if not original_init:
            # Try to find an __init__ in the bases
            for base in reversed(bases):
                if base.__init__ is not object.__init__:
                    original_init = base.__init__
                    break

        # We'll define a new __init__ only if there's an actual one to wrap
        if original_init:

            def __init__(self: object, *args: Any, **kwargs: Any) -> None:
                if type(self) is cls:
                    assert original_init is not None, "No valid __init__ found!"
                    sig = inspect.signature(original_init)
                    bound = sig.bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    all_args = dict(bound.arguments)
                    all_args.pop("self", None)
                    self._init_args = all_args  # type: ignore[attr-defined]

                original_init(self, *args, **kwargs)

        # The rest is basically the same
        # But we must create the class first so we can set new_init.__signature__ = ...
        cls = super().__new__(mcs, name, bases, namespace)

        # If we actually did define new_init, attach it
        if original_init:
            # Replace/attach the new __init__ to cls
            setattr(cls, "__init__", __init__)

            # Manually override the function's signature
            init_sig = inspect.signature(original_init)
            __init__.__signature__ = init_sig  # type: ignore[attr-defined]

        original_repr = namespace.get("__repr__")

        if not original_repr:
            # Try to find an __repr__ in the bases
            for base in reversed(bases):
                if base.__repr__ is not object.__repr__:
                    original_repr = base.__repr__
                    break

        # If no user-defined __repr__, attach a default
        if not original_repr or getattr(
            original_repr, "__signature_repr_generated__", False
        ):

            def __repr__(self: object) -> str:
                # Case 1: No real __init__ found at all => just return ClassName()
                if original_init is None:
                    return f"{name}()"

                # Case 2: If the class or its parents do define an __init__, we check
                # for _init_args. If the class isn't wrapping init, your class would
                # have to set _init_args itself (or you'll just get a normal object repr).
                if not hasattr(self, "_init_args"):
                    return super(type(self), self).__repr__()

                # Build param=value only if they differ from default
                sig = inspect.signature(original_init)
                parts = []
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    default = param.default
                    if (
                        param_name in self._init_args
                        and self._init_args[param_name] != default
                    ):
                        parts.append(
                            f"{param_name}={repr(self._init_args[param_name])}"
                        )
                return f"{name}({', '.join(parts)})"

            # Tag it so children know it's auto-generated
            __repr__.__signature_repr_generated__ = True  # type: ignore[attr-defined]

            setattr(cls, "__repr__", __repr__)

        return cls


class SignatureReprABCMeta(SignatureReprMeta, abc.ABCMeta):
    """
    A metaclass combining `SignatureReprMeta` and `ABCMeta`.

    This metaclass extends `SignatureReprMeta`, allowing abstract base classes (`ABC`) to inherit
    automatic argument capturing and dynamic `__repr__` generation.

    Usage:
    ```python
    class AbstractExample(metaclass=SignatureReprABCMeta):
        @abc.abstractmethod
        def some_method(self):
            pass

    class ConcreteExample(AbstractExample):
        def __init__(self, value, flag=True):
            self.value = value
            self.flag = flag

    obj = ConcreteExample(42)
    print(obj)  # Output: ConcreteExample(value=42) (flag is omitted since it uses the default True)
    ```
    """
