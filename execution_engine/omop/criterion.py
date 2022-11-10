from abc import ABC, abstractmethod

from .concepts import ConceptSet


class Criterion(ABC):
    """A criterion in a cohort definition."""

    def __init__(self, concept_set: ConceptSet):  # fixme: proper import / type hint
        self._concept_set = concept_set

    @property
    def type(self) -> str:
        """
        Get the type of the criterion.
        """
        return self.__class__.__name__

    @property
    def concept_set(self) -> ConceptSet:
        """The concept set used by the criterion."""
        return self._concept_set

    @abstractmethod
    def sql(self, table_in: str | None, table_out: str) -> str:
        """
        Get the SQL representation of the criterion.
        """
        raise NotImplementedError()


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
