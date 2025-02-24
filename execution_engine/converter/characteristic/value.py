from abc import ABC
from typing import Type

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.constants import CohortCategory
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.criterion import parse_code, parse_value
from execution_engine.omop.criterion.concept import ConceptCriterion


class AbstractValueCharacteristic(AbstractCharacteristic, ABC):
    """An abstract characteristic that is not only defined by a concept but has additionally a value."""

    _criterion_class: Type[ConceptCriterion]
    _concept_value_static = False

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

        c: AbstractCharacteristic = cls(exclude=characteristic.exclude)
        c.type = type_omop_concept
        c.value = value

        return c

    def to_positive_criterion(self) -> ConceptCriterion:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            category=CohortCategory.POPULATION,
            concept=self.type,
            value=self.value,
            static=self._concept_value_static,
        )
