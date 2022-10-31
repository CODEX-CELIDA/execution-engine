from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterator, List, Optional, Tuple, Union

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.evidencevariable import (
    EvidenceVariableCharacteristicDefinitionByTypeAndValue,
)

from . import SNOMEDCT
from .fhir import isLOINC, isSNOMEDCT, tx_client
from .omop import webapi
from .omop.concepts import Concept, ConceptSet
from .omop.criterion import ConditionOccurrence, Criterion
from .omop.vocabulary import Vocabulary


def get_coding(cc: CodeableConcept) -> Coding:
    """
    Get the first (and only one) coding from a CodeableConcept.
    """
    assert len(cc.coding) == 1, "CodeableConcept must have exactly one coding"
    return cc.coding[0]


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

    def __init__(self, code: Code, threshold: Optional[int] = None) -> None:
        """
        Creates a new characteristic combination.
        """
        self.code: CharacteristicCombination.Code = code
        self.characteristics: List[
            Union["AbstractCharacteristic", "CharacteristicCombination"]
        ] = []
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
    """

    def __init__(self) -> None:
        self._value: Optional[Concept] = None

    @classmethod
    @abstractmethod
    def from_fhir(
        cls, char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue
    ) -> "AbstractCharacteristic":
        """Creates a new characteristic from a FHIR EvidenceVariable."""
        raise NotImplementedError()

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
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the given FHIR EvidenceVariable is a valid characteristic."""
        raise NotImplementedError()

    @abstractmethod
    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Convert this characteristic to an OMOP criterion."""
        raise NotImplementedError()


class CharacteristicFactory:
    """Factory for creating characteristics objects from EvidenceVariable.characteristic."""

    def __init__(self) -> None:
        self._characteristic_types: List[AbstractCharacteristic] = []

    def register_characteristic_type(
        self, characteristic: AbstractCharacteristic
    ) -> None:
        """Register a new characteristic type."""
        self._characteristic_types.append(characteristic)

    def get_characteristic(self, fhir: CodeableConcept) -> AbstractCharacteristic:
        """Get the characteristic type for the given CodeableConcept from EvidenceVariable.characteristic.definitionByTypeAndValue.type."""
        for characteristic in self._characteristic_types:
            if characteristic.valid(fhir):
                return characteristic.from_fhir(fhir)
        raise ValueError("No characteristic type matched the FHIR definition.")


SCT_CLINICAL_FINDING = "404684003"  # Clinical finding (finding)
SCT_ALLERGIC_DISPOSITION = "609328004"  # Allergic disposition (finding)
SCT_RADIOLOGIC_FINDING = "118247008"  # Radiologic finding (finding)
SCT_EPISODE_OF_CARE_TYPE = "78030-4"  # Episode of care Type
SCT_PROCEDURE = "71388002"  # Procedure (procedure)
SCT_VENTILATOR_OBSERVABLE = "364698001"  # Ventilator observable (observable entity)

VS_VENTILATOR_OBSERVABLE = "https://medizininformatik-initiative.de/fhir/ext/modul-icu/ValueSet/Code-Observation-Beatmung-LOINC"


class ConditionCharacteristic(AbstractCharacteristic):
    """A condition characteristic  in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the given FHIR definition is a valid condition characteristic in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        return isSNOMEDCT(cc) and cc.code == SCT_CLINICAL_FINDING

    @classmethod
    def from_fhir(
        cls, char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue
    ) -> AbstractCharacteristic:
        """Creates a new ConditionCharacteristic from a FHIR EvidenceVariable.characteristic."""
        assert cls.valid(char_definition), "Invalid characteristic definition"

        cc = get_coding(char_definition.valueCodeableConcept)
        omop_concept = webapi.get_standard_concept(
            Vocabulary.from_url(cc.system), cc.code
        )

        c = ConditionCharacteristic()
        c.value = omop_concept

        return c

    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Converts this characteristic to an OMOP criterion of type ConditionOccurrence."""
        cs = ConceptSet(name=self.value.name, concept=self.value)  # fixme
        return cs, ConditionOccurrence(cs)


class AllergyCharacteristic(ConditionCharacteristic):
    """An allergy characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the given characteristic definition is a valid allergy characteristic in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        return isSNOMEDCT(cc) and cc.code == SCT_ALLERGIC_DISPOSITION


class RadiologyCharacteristic(AbstractCharacteristic):
    """A radiology characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the characteristic is a radiology finding in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        return isSNOMEDCT(cc) and cc.code == SCT_RADIOLOGIC_FINDING


class ProcedureCharacteristic(AbstractCharacteristic):
    """A procedure characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the given characteristic definition is a procedure characteristic in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        return isSNOMEDCT(cc) and cc.code == SCT_PROCEDURE


class EpisodeOfCareCharacteristic(AbstractCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if the given characteristic definition is a valid episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        return isLOINC(cc) and cc.code == SCT_EPISODE_OF_CARE_TYPE


class VentilationObservableCharacteristic(AbstractCharacteristic):
    """A ventilation observable characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if characteristic is a ventilation observable in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.type)
        ventilationObservablesSCT = tx_client.get_descendents(
            SNOMEDCT, SCT_VENTILATOR_OBSERVABLE
        )
        ventilationObservablesLOINC = tx_client.get_value_set(VS_VENTILATOR_OBSERVABLE)

        return (isSNOMEDCT(cc) and cc.code in ventilationObservablesSCT) or (
            isLOINC(cc) and cc.code in ventilationObservablesLOINC
        )


class LaboratoryCharacteristic(AbstractCharacteristic):
    """A laboratory characteristic in the context of CPG-on-EBM-on-FHIR."""

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristicDefinitionByTypeAndValue,
    ) -> bool:
        """Checks if characteristic is a laboratory observable in the context of CPG-on-EBM-on-FHIR"""
        cc = get_coding(char_definition.type)
        # TODO: Don't just use all LOINC codes, but restrict to subset of important ones (or actually used ones)
        return isLOINC(cc)
