from abc import ABC, abstractmethod
from typing import Any, Type

from fhir.resources.coding import Coding
from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.converter.converter import CriterionConverter
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.vocabulary import standard_vocabulary
from execution_engine.util.value import Value


class AbstractCharacteristic(CriterionConverter, ABC):
    """
    An abstract characteristic.

    An instance of this class represents a single characteristic entry of the EvidenceVariable resource
    in the context of CPG-on-EBM-on-FHIR. In the Implementation Guide (specifically, the EligibilityCriteria profile),
    several types of characteristics are defined, including:
    - Condition
    - Allergy
    - Radiologic finding
    - Episode of Care
    - Procedure
    - Ventilation Observable
    - Laboratory Value
    Each of these slices from the Implementation Guide is represented by a subclass of this class.

    Subclasses must define the following methods:
    - valid: returns True if the supplied characteristic falls within the scope of the subclass
    - from_fhir: creates a new instance of the subclass from a FHIR EvidenceVariable.characteristic element
    - to_criterion(): converts the characteristic to a Criterion
    """

    _criterion_class: Type[Criterion]
    _type: Concept
    _value: Value

    def __init__(self, exclude: bool) -> None:
        super().__init__(exclude=exclude)

    @classmethod
    @abstractmethod
    def from_fhir(
        cls, characteristic: EvidenceVariableCharacteristic
    ) -> "AbstractCharacteristic":
        """Creates a new characteristic from a FHIR EvidenceVariable."""
        raise NotImplementedError()

    @property
    def exclude(self) -> bool:
        """Returns True if this characteristic is an exclusion."""
        # if exclude is not set in the FHIR resource, it defaults to False
        return self._exclude

    @property
    def type(self) -> Concept:
        """The type of this characteristic."""
        return self._type

    @type.setter
    def type(self, type: Concept) -> None:
        """Sets the type of this characteristic."""
        self._type = type

    @property
    def value(self) -> Any:
        """The value of this characteristic."""
        return self._value

    @value.setter
    def value(self, value: Value) -> None:
        """Sets the value of this characteristic."""
        self._value = value

    @staticmethod
    @abstractmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if the given FHIR EvidenceVariable is a valid characteristic."""
        raise NotImplementedError()

    @staticmethod
    def get_standard_concept(cc: Coding) -> Concept:
        """
        Gets the standard concept for a CodeableConcept.Coding.
        """
        return standard_vocabulary.get_standard_concept(cc.system, cc.code)

    @staticmethod
    def get_concept(cc: Coding, standard: bool = True) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return standard_vocabulary.get_concept(cc.system, cc.code, standard=standard)

    @abstractmethod
    def to_positive_criterion(self) -> Criterion:
        """
        Converts this characteristic to a "Positive" Criterion.

        Positive criterion means that a possible excluded flag is disregarded. Instead, the exclusion
        is later introduced (in the to_criterion() method) via a LogicalCriterionCombination.Not).
        """
        raise NotImplementedError()
