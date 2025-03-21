import importlib
import os
import sys
import types
from collections import namedtuple
from typing import TypeVar


def get_processing_module(
    name: str = "rectangle", version: str = "auto"
) -> types.ModuleType:
    """
    Returns the processing module with the given name.

    Available processing modules:
        - rectangle (faster, using rectangles intersection/union)

    :param name: name of the processing module
    :param version: version of the processing module
    """

    if name == "rectangle":
        if version not in ["auto", "python", "cython"]:
            raise ValueError("Unknown processing module version: {}".format(version))

        os.environ["PROCESS_RECTANGLE_VERSION"] = version

        module_path = "execution_engine.task.process.rectangle"
        if module_path in sys.modules:
            # The module is in sys.modules, reload it
            rectangle_module = importlib.reload(sys.modules[module_path])
        else:
            # Import the module for the first time
            rectangle_module = importlib.import_module(module_path)

        return rectangle_module
    else:
        raise ValueError("Unknown processing module: {}".format(name))


Interval = namedtuple("Interval", ["lower", "upper", "type"])
IntervalWithCount = namedtuple("IntervalWithCount", ["lower", "upper", "type", "count"])

AnyInterval = Interval | IntervalWithCount
GeneralizedInterval = None | AnyInterval

TInterval = TypeVar("TInterval", bound=AnyInterval)


def interval_like(interval: TInterval, start: int, end: int) -> TInterval:
    """
    Return a copy of the given interval with its lower and upper bounds replaced.

    Args:
        interval (I): The interval to copy. Must be one of Interval or IntervalWithCount.
        start (datetime): The new lower bound.
        end (datetime): The new upper bound.

    Returns:
        I: A copy of the interval with updated lower and upper bounds.
    """

    return interval._replace(lower=start, upper=end)  # type: ignore[return-value]
