from abc import ABC, abstractmethod
from typing import Callable, cast

from fhir.resources.element import Element
from fhir.resources.evidencevariable import EvidenceVariableCharacteristicTimeFromEvent

from execution_engine.converter.criterion import parse_value
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util import logic
from execution_engine.util.value import ValueNumeric


def _wrap_criteria_with_factory(
    combo: logic.BooleanFunction,
    factory: Callable[[logic.BaseExpr], logic.TemporalCount],
) -> logic.BooleanFunction:
    """
    Recursively wraps all Criterion instances within a combination using the factory.
    """
    children = []
    # Loop through all elements
    for element in combo.args:
        if isinstance(element, logic.BooleanFunction):
            # Recursively wrap nested combinations
            children.append(_wrap_criteria_with_factory(element, factory))
        elif isinstance(element, Criterion):
            # Wrap individual criteria with the factory
            children.append(factory(element))
        else:
            raise ValueError(f"Unexpected element type: {type(element)}")

    # Create a new combination of the same type with the same operator
    return combo.__class__(children)


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
    def to_temporal_combination(self, combo: logic.BaseExpr) -> logic.TemporalCount:
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
    def to_temporal_combination(self, combo: logic.BaseExpr) -> logic.TemporalCount:
        """
        Wraps Criterion/CriterionCombinaion with a TemporalIndicatorCombination
        """
        raise NotImplementedError("must be implemented by class")
