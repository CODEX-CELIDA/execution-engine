from abc import abstractmethod
from typing import Callable, cast

from fhir.resources.evidencevariable import EvidenceVariableCharacteristicTimeFromEvent

from execution_engine.converter.criterion import parse_value
from execution_engine.converter.temporal_indicator import TemporalIndicator
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
    def valid(cls, fhir: EvidenceVariableCharacteristicTimeFromEvent) -> bool:
        """Checks if the given FHIR definition is a valid TemporalIndicator in the context of CPG-on-EBM-on-FHIR."""

        assert isinstance(
            fhir, EvidenceVariableCharacteristicTimeFromEvent
        ), f"Expected timeFromEvent type, got {fhir.__class__.__name__}"

        cc = get_coding(fhir.eventCodeableConcept)

        return cls._event_vocabulary.is_system(cc.system) and cc.code == cls._event_code

    @classmethod
    def from_fhir(
        cls, fhir: EvidenceVariableCharacteristicTimeFromEvent
    ) -> "TemporalIndicator":
        """
        Creates a new TemporalIndicator from a FHIR PlanDefinition.
        """
        assert isinstance(
            fhir, EvidenceVariableCharacteristicTimeFromEvent
        ), f"Expected timeFromEvent type, got {fhir.__class__.__name__}"

        value = None

        if fhir.range is not None and fhir.quantity is not None:
            raise ValueError(
                "Must specify either Range or Quantity in characteristic.timeFromEvent, not both."
            )

        if fhir.range is not None:
            value = cast(ValueNumeric, parse_value(fhir.range))

        if fhir.quantity is not None:
            value = cast(ValueNumeric, parse_value(fhir.quantity))

        return cls(value)

    @abstractmethod
    def to_interval_criterion(self) -> logic.BaseExpr:
        """
        Returns the criterion that returns the intervals during the enclosed criterion/combination is evaluated.

        This criterion comes from a "timeFromEvent" field in EvidenceVariable.characteristic and therefore
        specifies some time window (a.k.a. interval) during which the actual characteristic is supposed to happen.
        For example, the characteristic could be some kind of measurement to be performed, and the timeFromEvent
        could be "post surgical".

        The interval criterion returned by the TimeFromEvent class is later AND-combined (if there are more than one
        timeFromEvent requirements defined - they are always supposed to be simultaneously fulfilled, i.e. AND-combined)
        - and a logic.Presence TemporalIndicator is instantiated, with the AND-combination of interval criteria, and
        each single criterion contained in the characteristic to which this timeFromEvent belongs is wrapped with that
        logic.Presence( *args, interval_criterion=interval_criterion).


        Note that it is not the (potential) combination of single criteria in characteristic (if characteristic
        contains a definitionByCombination element) that is wrapped with logic.Presence, but the single criteria,
        because e.g. AND-combining single measurements to be performed would likely result in no positive intervals,
        because measurements are not performed simultaneously.
        """
        raise NotImplementedError("must be implemented by class")
