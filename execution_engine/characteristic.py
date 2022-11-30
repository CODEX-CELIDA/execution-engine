from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterator, Type, Union

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
from .omop.concepts import Concept
from .omop.criterion import (
    ConceptCriterion,
    ConditionOccurrence,
    Criterion,
    Measurement,
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
from .util import Value, ValueConcept, ValueNumber


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

    def __init__(self, code: Code, exclude: bool, threshold: int | None = None) -> None:
        """
        Creates a new characteristic combination.
        """
        self.code: CharacteristicCombination.Code = code
        self.characteristics: list[
            Union["AbstractCharacteristic", "CharacteristicCombination"]
        ] = []
        self.exclude: bool = exclude
        self.threshold: int | None = threshold

    def add(
        self,
        characteristic: Union["AbstractCharacteristic", "CharacteristicCombination"],
    ) -> None:
        """Adds a characteristic to this combination."""
        self.characteristics.append(characteristic)

    def __iter__(self) -> Iterator:
        """Return an iterator for the characteristics of this combination."""
        return iter(self.characteristics)


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


class CharacteristicFactory:
    """Factory for creating characteristics objects from EvidenceVariable.characteristic."""

    def __init__(self) -> None:
        self._characteristic_types: list[Type[AbstractCharacteristic]] = []

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

        c = cls(characteristic.exclude)
        c.value = omop_concept

        return c

    def to_criterion(self) -> Criterion:
        """Converts this characteristic to a Criterion."""
        return self._criterion_class(
            name=self.value.name, exclude=self.exclude, concept=self.value, value=None
        )


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


class EpisodeOfCareCharacteristic(AbstractCodeableConceptCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = LOINC_EPISODE_OF_CARE_TYPE
    _concept_vocabulary = LOINC
    _criterion_class = VisitOccurrence


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
