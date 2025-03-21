from abc import ABC, abstractmethod

from fhir.resources.element import Element

from execution_engine.util import logic


class TemporalIndicator(ABC):
    """
    EvidenceVariable.characteristic.timeFromEvent in the context of CPG-on-EBM-on-FHIR.
    """

    @classmethod
    @abstractmethod
    def valid(cls, fhir: Element) -> bool:
        """Checks if the given FHIR definition is a valid TemporalIndicator in the context of CPG-on-EBM-on-FHIR."""
        raise NotImplementedError("must be implemented by class")

    @classmethod
    @abstractmethod
    def from_fhir(cls, fhir: Element) -> "TemporalIndicator":
        """
        Creates a new TemporalIndicator from a FHIR PlanDefinition.
        """
        raise NotImplementedError("must be implemented by class")

    @abstractmethod
    def to_interval_criterion(self) -> logic.BaseExpr:
        """
        Returns the criterion that returns the intervals during the enclosed criterion/combination is evaluated.

        This criterion (in FHIR) specifies some time window (a.k.a. interval) during which the actual criterion or
        combination of criteria are supposed to happen. For example, the criterion could be some kind of measurement
        to be performed, and the temporal indicator could be "post surgical".

        The interval criterion returned by this class is later AND-combined (if there are more than one
        temporal requirements defined - they are always supposed to be simultaneously fulfilled, i.e. AND-combined)
        - and a logic.Presence TemporalIndicator is instantiated, with the AND-combination of interval criteria, and
        each single criterion contained in the characteristic to which this timeFromEvent belongs is wrapped with that
        logic.Presence( *args, interval_criterion=interval_criterion).

        Note that it is not the (potential) combination of single criteria that is wrapped with logic.Presence, but the
        single criteria, because e.g. AND-combining single measurements to be performed would likely result in no
        positive intervals, because measurements are not performed simultaneously.
        """
        raise NotImplementedError("must be implemented by class")
