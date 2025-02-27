from abc import ABC, abstractmethod
from typing import Callable, cast

from fhir.resources.element import Element
from fhir.resources.evidencevariable import EvidenceVariableCharacteristicTimeFromEvent

from execution_engine.converter.criterion import parse_value
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.combination.temporal import (
    TemporalIndicatorCombination,
)
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util.value import ValueNumeric


def _wrap_criteria_with_factory(
    combo: CriterionCombination,
    factory: Callable[[Criterion | CriterionCombination], TemporalIndicatorCombination],
) -> CriterionCombination:
    """
    Recursively wraps all Criterion instances within a combination using the factory.
    """
    # Create a new combination of the same type with the same operator
    new_combo = combo.__class__(operator=combo.operator)

    # Loop through all elements
    for element in combo:
        if isinstance(element, LogicalCriterionCombination):
            # Recursively wrap nested combinations
            new_combo.add(_wrap_criteria_with_factory(element, factory))
        elif isinstance(element, Criterion):
            # Wrap individual criteria with the factory
            new_combo.add(factory(element))
        else:
            raise ValueError(f"Unexpected element type: {type(element)}")

    return new_combo


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
        self, combo: Criterion | CriterionCombination
    ) -> CriterionCombination:
        """
        Wraps Criterion/CriterionCombinaion with a TemporalIndicatorCombination
        """
        raise NotImplementedError("must be implemented by class")


class TimeFromEvent(TemporalIndicator):
    """
    EvidenceVariable.characteristic.timeFromEvent in the context of CPG-on-EBM-on-FHIR.
    """

    _event_vocabulary: type[AbstractVocabulary]
    _event_code: str
    _value: ValueNumeric | None

    def __init__(
        self,
        value: ValueNumeric | None,
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

        value = None

        if tfe.range is not None and tfe.quantity is not None:
            raise ValueError(
                "Must specify either Range or Quantity in characteristic.timeFromEvent, not both."
            )

        if tfe.range is not None:
            value = cast(ValueNumeric, parse_value(tfe.range))

        if tfe.quantity is not None:
            value = cast(ValueNumeric, parse_value(tfe.quantity))

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
        self, combo: Criterion | CriterionCombination
    ) -> CriterionCombination:
        """
        Wraps Criterion/CriterionCombinaion with a TemporalIndicatorCombination
        """
        raise NotImplementedError("must be implemented by class")
