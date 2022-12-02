from typing import Type

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from .abstract import AbstractCharacteristic


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
