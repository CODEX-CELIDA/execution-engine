from typing import Dict

from .concepts import ConceptSet


class Criterion:
    """A criterion in a cohort definition."""

    def __init__(self, concept_set: ConceptSet):
        self._concept_set = concept_set

    def getType(self) -> str:
        """
        Get the type of the criterion.
        """
        return self.__class__.__name__

    @property
    def concept_set(self) -> ConceptSet:
        """The concept set used by the criterion."""
        return self._concept_set

    def json(self) -> Dict:
        """Return the JSON representation of the criterion."""

        assert self._concept_set.id is not None, "Concept set ID is None"

        return {
            self.getType(): {
                "CodesetId": self._concept_set.id,
            }
        }


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


class Visit(Criterion):
    """A visit criterion in a cohort definition."""

    pass
