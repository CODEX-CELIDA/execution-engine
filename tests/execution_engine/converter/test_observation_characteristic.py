# loinc 48066-5
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


def test_observation_loinc():
    element = EvidenceVariableCharacteristic()
    element.definitionByTypeAndValue = (
        EvidenceVariableCharacteristicDefinitionByTypeAndValue(
            type=CodeableConcept(
                coding=[Coding(system="http://loinc.org", code="48066-5")]
            ),
            valueQuantity=Quantity(value=2),
        )
    )

    ret = ObservationCharacteristic.valid(element)

    assert ret is True
