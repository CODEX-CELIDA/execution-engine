from abc import ABC
from typing import Type

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.evidencevariable import EvidenceVariableCharacteristic
from fhir.resources.quantity import Quantity
from fhir.resources.range import Range

from ...fhir.util import get_coding
from ...omop.criterion.concept import ConceptCriterion
from ...util import ValueConcept, ValueNumber
from .abstract import AbstractCharacteristic


class AbstractValueCharacteristic(AbstractCharacteristic, ABC):
    """An abstract characteristic that is not only defined by a concept but has additionally a value."""

    _criterion_class: Type[ConceptCriterion]

    @classmethod
    def from_fhir(
        cls, characteristic: EvidenceVariableCharacteristic
    ) -> AbstractCharacteristic:
        """Creates a new Characteristic instance from a FHIR EvidenceVariable.characteristic."""
        assert cls.valid(characteristic), "Invalid characteristic definition"

        cc = get_coding(characteristic.definitionByTypeAndValue.type)
        type_omop_concept = cls.get_standard_concept(cc)

        c: AbstractCharacteristic = cls(characteristic.exclude)
        c.type = type_omop_concept

        value = c.select_value(characteristic.definitionByTypeAndValue)
        if isinstance(value, CodeableConcept):
            cc = get_coding(value)
            value_omop_concept = cls.get_standard_concept(cc)
            c.value = ValueConcept(value=value_omop_concept)
        elif isinstance(value, Quantity):
            c.value = ValueNumber(
                value=value.value, unit=cls.get_standard_concept_unit(value.unit)
            )
        elif isinstance(value, Range):

            if value.low is not None and value.high is not None:
                assert (
                    value.low.code == value.high.code
                ), "Range low and high unit must be the same"

            unit_code = value.low.code if value.low is not None else value.high.code

            def value_or_none(x: Quantity | None) -> float | None:
                if x is None:
                    return None
                return x.value

            c.value = ValueNumber(
                unit=cls.get_standard_concept_unit(unit_code),
                value_min=value_or_none(value.low),
                value_max=value_or_none(value.high),
            )

        else:
            raise NotImplementedError

        return c

    def to_criterion(self) -> ConceptCriterion:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            name=self.type.name,
            exclude=self.exclude,
            concept=self.type,
            value=self.value,
        )
