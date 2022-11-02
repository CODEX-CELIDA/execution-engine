from typing import Dict, Optional

from .cohort_definition.value import AbstractValue
from .concepts import ConceptSet


class Criterion:
    """A criterion in a cohort definition."""

    def __init__(
        self, concept_set: ConceptSet, value: Optional[AbstractValue] = None
    ):  # fixme: proper import / type hint
        self._concept_set = concept_set
        self._value = value

    @property
    def type(self) -> str:
        """
        Get the type of the criterion.
        """
        return self.__class__.__name__

    @property
    def value(self) -> Optional[AbstractValue]:
        """Get the value of the criterion."""
        return self._value

    @value.setter
    def value(self, value: AbstractValue) -> None:
        self._value = value

    @property
    def concept_set(self) -> ConceptSet:
        """The concept set used by the criterion."""
        return self._concept_set

    def json(self) -> Dict:
        """Return the JSON representation of the criterion."""

        assert self._concept_set.id is not None, "Concept set ID is None"

        json = {
            self.type: {
                "CodesetId": self._concept_set.id,
            }
        }

        if self.value is not None:
            json[self.type] |= self.value.json()

        return json


class ConditionOccurrence(Criterion):
    """A condition occurrence criterion in a cohort definition."""

    pass


class DrugExposure(Criterion):
    """A drug exposure criterion in a cohort definition."""

    pass


class Measurement(Criterion):
    """A measurement criterion in a cohort definition."""

    pass


class Observation(Criterion):
    """An observation criterion in a cohort definition."""

    pass


class ProcedureOccurrence(Criterion):
    """A procedure occurrence criterion in a cohort definition."""

    pass


class VisitOccurrence(Criterion):
    """A visit criterion in a cohort definition."""

    pass
