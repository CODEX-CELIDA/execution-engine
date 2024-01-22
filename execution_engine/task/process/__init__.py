import types
from collections import namedtuple


def get_processing_module(name: str = "rectangle") -> types.ModuleType:
    """
    Returns the processing module with the given name.

    Available processing modules:
        - rectangle (faster, using rectangles intersection/union)
        - interval_portion (slower, using Portion Interval)

    :param name: name of the processing module
    """
    if name == "interval_portion":
        from execution_engine.task.process import interval_portion

        return interval_portion
    elif name == "rectangle":
        from execution_engine.task.process import rectangle

        return rectangle
    else:
        raise ValueError("Unknown processing module: {}".format(name))


Interval = namedtuple("Interval", ["lower", "upper", "type"])
