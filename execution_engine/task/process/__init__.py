import importlib
import os
import sys
import types
from collections import namedtuple


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
IntervalWithTypeCounts = namedtuple(
    "IntervalWithTypeCounts", ["lower", "upper", "counts"]
)
