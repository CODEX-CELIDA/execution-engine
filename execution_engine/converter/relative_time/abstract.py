from abc import abstractmethod
from typing import cast

from fhir.resources.extension import Extension

from execution_engine.converter.criterion import parse_value
from execution_engine.converter.temporal_indicator import TemporalIndicator
from execution_engine.fhir.util import get_coding, get_extension
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util import logic
from execution_engine.util.value import ValueNumeric


class RelativeTime(TemporalIndicator):
    """
    extension[relativeTime] in the context of CPG-on-EBM-on-FHIR.
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
    def valid(cls, fhir: Extension) -> bool:
        """Checks if the given FHIR definition is a valid TemporalIndicator in the context of CPG-on-EBM-on-FHIR."""

        assert isinstance(
            fhir, Extension
        ), f"Expected Extension type, got {fhir.__class__.__name__}"

        context = get_extension(fhir, "contextCode")

        if not context:
            raise ValueError("Required relativeTime:contextCode not found")

        cc = get_coding(context.valueCodeableConcept)

        return cls._event_vocabulary.is_system(cc.system) and cc.code == cls._event_code

    @classmethod
    def from_fhir(cls, fhir: Extension) -> "TemporalIndicator":
        """
        Creates a new TemporalIndicator from a FHIR PlanDefinition.
        """
        assert isinstance(
            fhir, Extension
        ), f"Expected Extension type, got {fhir.__class__.__name__}"

        value = None

        offset = get_extension(fhir, "offset")

        if offset:
            value = cast(ValueNumeric, parse_value(offset.valueDuration))

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
