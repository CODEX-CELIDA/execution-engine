from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

from ..concepts import Concept


class AbstractValue(ABC):
    """
    A value for a criterion.
    """

    @abstractmethod
    def json(self) -> Dict:
        """Return the JSON representation of the value."""
        pass


class ValueNumber(AbstractValue):
    """
    A value for a criterion of type number.
    """

    class Operator(Enum):
        """
        The comparison operator for a number value.
        """

        LESS_THAN = "lt"
        LESS_OR_EQUAL_TO = "lte"
        EQUAL_TO = "eq"
        GREATER_THAN = "gt"
        GREATER_OR_EQUAL_TO = "gte"
        BETWEEN = "bt"
        NOT_BETWEEN = "!bt"

    def __init__(
        self, value: float, operator: Operator, unit: Concept, extent: Optional[float]
    ):
        self.value = value
        self.operator = operator
        self.unit = unit
        self.extent = extent

    def json(self) -> Dict:
        """
        Return the JSON representation of the value."""
        val = {
            "Value": self.value,
            "Op": self.operator.value,
        }
        if self.operator in [
            ValueNumber.Operator.BETWEEN,
            ValueNumber.Operator.NOT_BETWEEN,
        ]:
            assert self.extent is not None, "Extent must be set for 'between' operators"
            val["Extent"] = self.extent

        unit = [self.unit.json()]

        return {"ValueAsNumber": val, "Unit": unit}


class ValueConcept(AbstractValue):
    """
    A value for a criterion of type concept (from a vocabulary).
    """

    def __init__(self, concept: Concept):
        self.concept = concept

    def json(self) -> Dict:
        """
        Return the JSON representation of the value.
        """
        return {"ValueAsConcept": [self.concept.json()]}
