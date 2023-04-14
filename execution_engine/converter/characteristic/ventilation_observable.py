from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.clients import tx_client
from execution_engine.constants import (
    SCT_VENTILATOR_OBSERVABLE,
    VS_VENTILATOR_OBSERVATIONS,
)
from execution_engine.converter.characteristic.value import AbstractValueCharacteristic
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.vocabulary import LOINC, SNOMEDCT


class VentilationObservableCharacteristic(AbstractValueCharacteristic):
    """A ventilation observable characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement
    _concept_value_static = False

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a ventilation observable in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)

        ventilationObservablesSCT = tx_client.get_descendents(
            SNOMEDCT.system_uri, SCT_VENTILATOR_OBSERVABLE
        )

        # todo: needs to provided to the server because it is not an official value set
        ventilationObservablesLOINC = tx_client.get_value_set(
            VS_VENTILATOR_OBSERVATIONS
        )

        return (
            SNOMEDCT.is_system(cc.system) and cc.code in ventilationObservablesSCT
        ) or (LOINC.is_system(cc.system) and cc.code in ventilationObservablesLOINC)
