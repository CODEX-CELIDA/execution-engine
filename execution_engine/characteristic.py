from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, List, Optional, Tuple, Type, Union

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.evidencevariable import (
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicDefinitionByTypeAndValue,
)
from fhir.resources.quantity import Quantity
from fhir.resources.range import Range

from .clients import tx_client
from .constants import *
from .fhir.util import get_coding
from .omop.cohort_definition.value import AbstractValue, ValueConcept, ValueNumber
from .omop.concepts import Concept, ConceptSet
from .omop.criterion import (
    ConditionOccurrence,
    Criterion,
    Measurement,
    Observation,
    ProcedureOccurrence,
    VisitOccurrence,
)
from .omop.vocabulary import (
    LOINC,
    SNOMEDCT,
    UCUM,
    AbstractVocabulary,
    standard_vocabulary,
)


class CharacteristicCombination:
    """Combination of Characteristics"""

    class Code(Enum):
        """
        The code for the combination of characteristics.
        """

        ALL_OF = "all-of"  # all characteristics must be true
        ANY_OF = "any-of"  # at least one characteristic must be true
        AT_LEAST = "at-least"  # at least n characteristics must be true
        AT_MOST = "at-most"  # at most n characteristics must be true
        STATISTICAL = "statistical"  # statistical combination of characteristics
        NET_EFFECT = "net-effect"  # net effect of characteristics
        DATASET = "dataset"  # dataset of characteristics

    def __init__(
        self, code: Code, exclude: bool, threshold: Optional[int] = None
    ) -> None:
        """
        Creates a new characteristic combination.
        """
        self.code: CharacteristicCombination.Code = code
        self.characteristics: List[
            Union["AbstractCharacteristic", "CharacteristicCombination"]
        ] = []
        self.exclude: bool = exclude
        self.threshold: Optional[int] = threshold

    def add(
        self,
        characteristic: Union["AbstractCharacteristic", "CharacteristicCombination"],
    ) -> None:
        """Adds a characteristic to this combination."""
        self.characteristics.append(characteristic)

    def __iter__(self) -> Iterator:
        """Return an iterator for the characteristics of this combination."""
        return iter(self.characteristics)

    def to_omop(self) -> List[Tuple[ConceptSet, Criterion]]:
        """Convert this characteristic combination to a list of OMOP criteria."""
        return [c.to_omop() for c in self.characteristics]


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
    - to_omop: converts the characteristic to a list of OMOP criteria
    - from_fhir: creates a new instance of the subclass from a FHIR EvidenceVariable.characteristic element

    """

    _criterion_class: Type[Criterion]

    @dataclass
    class ValueNumber:
        """
        A value of type number.
        """

        unit: Concept
        value: Optional[float] = None
        value_min: Optional[float] = None
        value_max: Optional[float] = None

    def __init__(self, exclude: Optional[bool]) -> None:
        self._exclude = exclude
        self._type: Optional[Concept] = None
        self._value: Any = None

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
    def type(self) -> Optional[Concept]:
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

    @abstractmethod
    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Convert this characteristic to an OMOP criterion."""
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
        return standard_vocabulary.get_standard_concept(UCUM.system_uri, code)


class CharacteristicFactory:
    """Factory for creating characteristics objects from EvidenceVariable.characteristic."""

    def __init__(self) -> None:
        self._characteristic_types: List[Type[AbstractCharacteristic]] = []

    def register_characteristic_type(
        self, characteristic: Type[AbstractCharacteristic]
    ) -> None:
        """Register a new characteristic type."""
        self._characteristic_types.append(characteristic)

    def get_characteristic(
        self, fhir: EvidenceVariableCharacteristic
    ) -> AbstractCharacteristic:
        """Get the characteristic type for the given CodeableConcept from EvidenceVariable.characteristic.definitionByTypeAndValue.type."""
        for characteristic in self._characteristic_types:
            if characteristic.valid(fhir):
                return characteristic.from_fhir(fhir)
        raise ValueError("No characteristic type matched the FHIR definition.")


class AbstractCodeableConceptCharacteristic(AbstractCharacteristic):
    """
    An abstract characteristic that uses a CodeableConcept as its value.

    """  # fixme : docstring

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

        c = cls(characteristic.exclude)
        c.value = omop_concept

        return c

    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Converts this characteristic to an OMOP criterion of type ConditionOccurrence."""
        cs = ConceptSet(name=self.value.name, concept=self.value)  # fixme
        return cs, self._criterion_class(cs)


class ConditionCharacteristic(AbstractCodeableConceptCharacteristic):
    """A condition characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_CLINICAL_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence


class AllergyCharacteristic(AbstractCodeableConceptCharacteristic):
    """An allergy characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_ALLERGIC_DISPOSITION
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence


class RadiologyCharacteristic(AbstractCodeableConceptCharacteristic):
    """A radiology characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_RADIOLOGIC_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence


class ProcedureCharacteristic(AbstractCodeableConceptCharacteristic):
    """A procedure characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_PROCEDURE
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ProcedureOccurrence


class AbstractValueCharacteristic(AbstractCharacteristic, ABC):
    """An abstract characteristic that is not only defined by a concept but has additionally a value."""

    @classmethod
    def from_fhir(
        cls, characteristic: EvidenceVariableCharacteristic
    ) -> AbstractCharacteristic:
        """Creates a new Characteristic instance from a FHIR EvidenceVariable.characteristic."""
        assert cls.valid(characteristic), "Invalid characteristic definition"

        cc = get_coding(characteristic.definitionByTypeAndValue.type)
        type_omop_concept = cls.get_standard_concept(cc)

        c = cls(characteristic.exclude)
        c.type = type_omop_concept

        value = c.select_value(characteristic.definitionByTypeAndValue)
        if isinstance(value, CodeableConcept):
            cc = get_coding(value)
            value_omop_concept = cls.get_standard_concept(cc)
            c.value = value_omop_concept
        elif isinstance(value, Quantity):
            c.value = AbstractCharacteristic.ValueNumber(
                value=value.value, unit=cls.get_standard_concept_unit(value.unit)
            )
        elif isinstance(value, Range):

            if value.low is not None and value.high is not None:
                assert (
                    value.low.code == value.high.code
                ), "Range low and high unit must be the same"

            unit_code = value.low.code if value.low is not None else value.high.code

            c.value = AbstractCharacteristic.ValueNumber(
                unit=cls.get_standard_concept_unit(unit_code)
            )

            if value.low is not None:
                c.value.value_min = float(value.low.value)

            if value.high is not None:
                c.value.value_max = float(value.high.value)

        else:
            raise NotImplementedError

        return c

    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Converts this characteristic to an OMOP criterion."""

        assert self.type is not None, "Characteristic type must be set"

        cs = ConceptSet(name=self.type.name, concept=self.type)
        value: AbstractValue

        if isinstance(self.value, Concept):
            value = ValueConcept(self.value)
        elif isinstance(self.value, AbstractCharacteristic.ValueNumber):

            if self.value.value_min is not None and self.value.value_max is not None:
                operator = ValueNumber.Operator.BETWEEN
                value_num = self.value.value_min
                extent = self.value.value_max
            elif self.value.value_min is not None:
                operator = ValueNumber.Operator.GREATER_OR_EQUAL_TO
                value_num = self.value.value_min
                extent = None
            elif self.value.value_max is not None:
                operator = ValueNumber.Operator.LESS_OR_EQUAL_TO
                value_num = self.value.value_max
                extent = None
            elif self.value.value is not None:
                operator = ValueNumber.Operator.EQUAL_TO
                value_num = self.value.value
                extent = None
            else:
                raise ValueError("Value must have either value, or low or high range")

            value = ValueNumber(
                value=value_num, operator=operator, extent=extent, unit=self.value.unit
            )

        return cs, self._criterion_class(cs, value=value)


class EpisodeOfCareCharacteristic(AbstractCodeableConceptCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = LOINC_EPISODE_OF_CARE_TYPE
    _concept_vocabulary = LOINC
    _criterion_class = VisitOccurrence

    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Converts this characteristic to an OMOP criterion of type ConditionOccurrence."""
        return super().to_omop()


class LaboratoryCharacteristic(AbstractValueCharacteristic):
    """A laboratory characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a laboratory observable in the context of CPG-on-EBM-on-FHIR"""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)
        # TODO: Don't just use all LOINC codes, but restrict to subset of important ones (or actually used ones)
        return LOINC.is_system(cc.system)


class VentilationObservableCharacteristic(AbstractValueCharacteristic):
    """A ventilation observable characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a ventilation observable in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)
        ventilationObservablesSCT = tx_client.get_descendents(
            SNOMEDCT.system_uri, SCT_VENTILATOR_OBSERVABLE
        )
        ventilationObservablesLOINC = tx_client.get_value_set(VS_VENTILATOR_OBSERVABLE)

        return (
            SNOMEDCT.is_system(cc.system) and cc.code in ventilationObservablesSCT
        ) or (LOINC.is_system(cc.system) and cc.code in ventilationObservablesLOINC)
