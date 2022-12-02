from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from ...clients import tx_client
from ...constants import SCT_VENTILATOR_OBSERVABLE, VS_VENTILATOR_OBSERVABLE
from ...fhir.util import get_coding
from ...omop.criterion.measurement import Measurement
from ...omop.vocabulary import LOINC, SNOMEDCT
from .value import AbstractValueCharacteristic


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
