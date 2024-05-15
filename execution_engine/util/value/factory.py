from execution_engine.util.types import Dosage, Timing
from execution_engine.util.value.time import ValueDuration, ValuePeriod
from execution_engine.util.value.value import Value, ValueConcept, ValueNumber


def value_factory(class_name: str, data: dict) -> Value | Timing | Dosage:
    """
    Get a value object from a class name and data.

    :param class_name: The name of the class to instantiate.
    :param data: The data to pass to the class constructor.
    :return: The value object.
    :raises ValueError: If the class name is not recognized.
    """

    class_map = {
        "ValueNumber": ValueNumber,
        "ValueConcept": ValueConcept,
        "ValuePeriod": ValuePeriod,
        "ValueDuration": ValueDuration,
        "Dosage": Dosage,
        "Timing": Timing,
    }

    """Create a value from a dictionary."""
    if class_name not in class_map:
        raise ValueError(f"Unknown value class {class_name}")

    return class_map[class_name](**data)
