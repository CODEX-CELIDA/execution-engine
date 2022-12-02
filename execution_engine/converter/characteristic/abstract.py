from abc import ABC, abstractmethod
from typing import Any, Iterator, Type, Union

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.evidencevariable import (
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicDefinitionByTypeAndValue,
)
from fhir.resources.quantity import Quantity
from fhir.resources.range import Range

from execution_engine.clients import tx_client
from execution_engine.constants import *
from execution_engine.fhir.util import get_coding
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.criterion.visit_occurrence import VisitOccurrence
from execution_engine.omop.vocabulary import (
    LOINC,
    SNOMEDCT,
    UCUM,
    AbstractVocabulary,
    standard_vocabulary,
)
from execution_engine.util import Value, ValueConcept, ValueNumber


class AbstractCharacteristic(ABC):
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

    def __init__(self, exclude: bool | None) -> None:
        self._exclude = exclude
        self._type: Concept
        self._value: Value

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
        if self._exclude is None:
            return False
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
    def value(self, value: Concept) -> None:
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
    def select_value(
        c: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> Concept:
        """
        Selects the value of a characteristic by datatype.
        """
        if c.valueQuantity:
            return c.valueQuantity
        elif c.valueRange:
            return c.valueRange
        elif c.valueCodeableConcept:
            return c.valueCodeableConcept
        elif c.valueBoolean:
            return c.valueBoolean
        else:
            raise ValueError("No value found")

    @staticmethod
    def get_standard_concept(cc: Coding) -> Concept:
        """
        Gets the standard concept for a CodeableConcept.Coding.
        """
        return standard_vocabulary.get_standard_concept(cc.system, cc.code)

    @staticmethod
    def get_standard_concept_unit(code: str) -> Concept:
        """
        Gets the standard concept for a unit.
        """
        return standard_vocabulary.get_standard_unit_concept(code)

    @abstractmethod
    def to_criterion(self) -> Criterion:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()
