import requests
from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.clients import tx_client
from execution_engine.constants import (
    VS_VENTILATOR_OBSERVATIONS_MII_DOWNLOAD_URL,
    VS_VENTILATOR_OBSERVATIONS_SCT_DOWNLOAD_URL,
)
from execution_engine.converter.characteristic.value import AbstractValueCharacteristic
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.vocabulary import SNOMEDCT


class VentilationObservableCharacteristic(AbstractValueCharacteristic):
    """A ventilation observable characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement
    _concept_value_static = False

    _valueset = None

    @staticmethod
    def load_valueset_mii() -> dict:
        """
        Load the laboratory observation value set from the CPG-on-EBM-on-FHIR specification.
        """
        if VentilationObservableCharacteristic._valueset is None:
            VentilationObservableCharacteristic._valueset = requests.get(
                VS_VENTILATOR_OBSERVATIONS_MII_DOWNLOAD_URL,
                timeout=10,
            ).json()

        return VentilationObservableCharacteristic._valueset

    @staticmethod
    def load_valueset_sct() -> dict:
        """
        Load the laboratory observation value set from the CPG-on-EBM-on-FHIR specification.
        """
        if VentilationObservableCharacteristic._valueset is None:
            VentilationObservableCharacteristic._valueset = requests.get(
                VS_VENTILATOR_OBSERVATIONS_SCT_DOWNLOAD_URL,
                timeout=10,
            ).json()

        return VentilationObservableCharacteristic._valueset

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a ventilation observable in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)

        # gl 2024-07-08: disabled because now using valueset directly
        # ventilationObservablesSCT = tx_client.get_descendents(
        #     SNOMEDCT.system_uri, SCT_VENTILATOR_OBSERVABLE
        # )

        if SNOMEDCT.is_system(cc.system):
            valueset = VentilationObservableCharacteristic.load_valueset_sct()
        else:
            valueset = VentilationObservableCharacteristic.load_valueset_mii()

        return tx_client.code_in_valueset(valueset, cc.code, cc.system)
