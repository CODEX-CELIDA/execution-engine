# loinc 48066-5
import pytest
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.evidencevariable import (
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicDefinitionByTypeAndValue,
)
from fhir.resources.quantity import Quantity

from execution_engine.converter.characteristic.observation import (
    ObservationCharacteristic,
)


@pytest.mark.parametrize("code", ["48066-5", "3150-0"])
def test_observation_loinc(code: str):

    element = EvidenceVariableCharacteristic()
    element.definitionByTypeAndValue = (
        EvidenceVariableCharacteristicDefinitionByTypeAndValue(
            type=CodeableConcept(coding=[Coding(system="http://loinc.org", code=code)]),
            valueQuantity=Quantity(value=2),
        )
    )

    assert ObservationCharacteristic.valid(element) is True
