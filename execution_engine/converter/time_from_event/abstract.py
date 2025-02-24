from abc import ABC, abstractmethod

from fhir.resources.element import Element
from fhir.resources.evidencevariable import EvidenceVariableCharacteristicTimeFromEvent

from execution_engine.converter.criterion import parse_value
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.temporal import (
    TemporalIndicatorCombination,
)
from execution_engine.omop.vocabulary import AbstractStandardVocabulary
from execution_engine.util.value import Value


class TemporalIndicator(ABC):
    """
    EvidenceVariable.characteristic.timeFromEvent in the context of CPG-on-EBM-on-FHIR.
    """

    @classmethod
    @abstractmethod
    def from_fhir(cls, fhir: Element) -> "TemporalIndicator":
        """
        Creates a new TemporalIndicator from a FHIR PlanDefinition.
        """
        raise NotImplementedError("must be implemented by class")

    @classmethod
    @abstractmethod
    def valid(cls, fhir: Element) -> bool:
        """Checks if the given FHIR definition is a valid TemporalIndicator in the context of CPG-on-EBM-on-FHIR."""
        raise NotImplementedError("must be implemented by class")

    @abstractmethod
    def to_temporal_combination(
        self, mode: str
    ) -> TemporalIndicatorCombination | Criterion:
        """
        Converts the TemporalIndicator to a TemporalIndicatorCombination.
        """
        raise NotImplementedError("must be implemented by class")


class TimeFromEvent(TemporalIndicator):
    """
    EvidenceVariable.characteristic.timeFromEvent in the context of CPG-on-EBM-on-FHIR.
    """

    _event_vocabulary: AbstractStandardVocabulary
    _event_code: str
    _value: Value | None

    def __init__(
        self,
        value: Value | None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__()
        self._value = value

    @classmethod
    def from_fhir(cls, fhir: Element) -> "TemporalIndicator":
        """
        Creates a new TemporalIndicator from a FHIR PlanDefinition.
        """
        assert isinstance(
            fhir, EvidenceVariableCharacteristicTimeFromEvent
        ), f"Expected timeFromEvent type, got {fhir.__class__.__name__}"

        tfe: EvidenceVariableCharacteristicTimeFromEvent = fhir
        value = parse_value(tfe.range)

        return cls(value)

    @classmethod
    def valid(cls, fhir: Element) -> bool:
        """Checks if the given FHIR definition is a valid TemporalIndicator in the context of CPG-on-EBM-on-FHIR."""

        assert isinstance(
            fhir, EvidenceVariableCharacteristicTimeFromEvent
        ), f"Expected timeFromEvent type, got {fhir.__class__.__name__}"

        tfe: EvidenceVariableCharacteristicTimeFromEvent = fhir

        cc = get_coding(tfe.eventCodeableConcept)

        return cls._event_vocabulary.is_system(cc.system) and cc.code == cls._event_code

    @abstractmethod
    def to_temporal_combination(
        self, mode: str
    ) -> TemporalIndicatorCombination | Criterion:
        """
        Converts the TemporalIndicator to a TemporalIndicatorCombination or Criterion.
        """
        raise NotImplementedError("must be implemented by class")
