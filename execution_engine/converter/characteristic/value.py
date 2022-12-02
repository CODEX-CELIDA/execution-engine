from abc import ABC
from typing import Type

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from ...fhir.util import get_coding
from ...omop.criterion.concept import ConceptCriterion
from ..converter import parse_code, parse_value
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

        type_omop_concept = parse_code(characteristic.definitionByTypeAndValue.type)
        value = parse_value(
            value_parent=characteristic.definitionByTypeAndValue, value_prefix="value"
        )

        c: AbstractCharacteristic = cls(
            name=type_omop_concept.name, exclude=characteristic.exclude
        )
        c.type = type_omop_concept
        c.value = value

        return c

    def to_criterion(self) -> ConceptCriterion:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            name=self._name,
            exclude=self._exclude,
            concept=self.type,
            value=self.value,
        )
