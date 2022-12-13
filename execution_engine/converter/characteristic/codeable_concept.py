from typing import Type

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from ...fhir.util import get_coding
from ...omop.criterion.abstract import Criterion
from ...omop.criterion.concept import ConceptCriterion
from ...omop.vocabulary import AbstractVocabulary
from .abstract import AbstractCharacteristic


class AbstractCodeableConceptCharacteristic(AbstractCharacteristic):
    """
    A characteristic from FHIR EvidenceVariable that is defined by a valueCodeableConcept.

    This is a base class for characteristics that are defined by a valueCodeableConcept.
    Inheriting classes must define the _concept_code and _concept_vocabulary attributes,
    which are used to determine whether an EvidenceVariable.characteristic belongs
    to that class (by comparing characteristic.typeCodeableConcept to the attributes).

    The _criterion_class attribute is used to determine the type of criterion to create
    when converting the characteristic to a criterion. The concept specified in
    EvidenceVariable.characteristic.valueCodeableConcept is used to initialize the criterion.
    """

    _criterion_class: Type[ConceptCriterion]
    _concept_code: str
    _concept_vocabulary: Type[AbstractVocabulary]

    @classmethod
    def valid(
        cls,
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if the given FHIR definition is a valid condition characteristic in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )

    @classmethod
    def from_fhir(
        cls, characteristic: EvidenceVariableCharacteristic
    ) -> AbstractCharacteristic:
        """Creates a new Characteristic instance from a FHIR EvidenceVariable.characteristic."""
        assert cls.valid(characteristic), "Invalid characteristic definition"

        cc = get_coding(characteristic.definitionByTypeAndValue.valueCodeableConcept)
        omop_concept = cls.get_standard_concept(cc)

        c = cls(name=omop_concept.name, exclude=characteristic.exclude)
        c.value = omop_concept

        return c

    def to_criterion(self) -> Criterion:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            name=self._name,
            exclude=self._exclude,
            category="population",
            concept=self.value,
            value=None,
        )
