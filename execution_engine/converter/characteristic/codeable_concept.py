import logging
from typing import Type

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.fhir.util import get_coding
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util import logic


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

    """
    indicates the value of this concept can be considered constant during a health care encounter (e.g. weight, height,
    allergies, etc.) or if it is subject to change (e.g. laboratory values, vital signs, conditions, etc.)
    """
    _concept_value_static: bool

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
    def get_class_from_concept(
        cls, concept: Concept
    ) -> Type["AbstractCodeableConceptCharacteristic"]:
        """Gets the class that matches the given concept."""
        return cls

    @classmethod
    def from_fhir(
        cls, characteristic: EvidenceVariableCharacteristic
    ) -> AbstractCharacteristic:
        """Creates a new Characteristic instance from a FHIR EvidenceVariable.characteristic."""
        assert cls.valid(characteristic), "Invalid characteristic definition"

        cc = get_coding(characteristic.definitionByTypeAndValue.valueCodeableConcept)

        try:
            omop_concept = cls.get_standard_concept(cc)
        except ValueError:
            logging.warning(
                f"Concept {cc.code} not found in standard vocabulary {cc.system}. Using non-standard vocabulary."
            )
            omop_concept = cls.get_concept(cc, standard=False)

        class_ = cls.get_class_from_concept(omop_concept)

        c = class_(exclude=characteristic.exclude)
        c.value = omop_concept

        return c

    def to_positive_expression(self) -> logic.BaseExpr:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            concept=self.value,
            value=None,
            static=self._concept_value_static,
        )
