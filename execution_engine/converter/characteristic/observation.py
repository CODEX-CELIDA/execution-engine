import requests
from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.clients import tx_client
from execution_engine.constants import VS_LABORATORY_OBSERVATIONS_DOWNLOAD_URL
from execution_engine.converter.characteristic.value import AbstractValueCharacteristic
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.measurement import Measurement


class ObservationCharacteristic(AbstractValueCharacteristic):
    """An observation characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement
    _concept_value_static = False

    _valueset = None

    @staticmethod
    def load_valueset() -> dict:
        """
        Load the laboratory observation value set from the CPG-on-EBM-on-FHIR specification.
        """
        if ObservationCharacteristic._valueset is None:
            ObservationCharacteristic._valueset = requests.get(
                VS_LABORATORY_OBSERVATIONS_DOWNLOAD_URL,
                timeout=10,
            ).json()

        return ObservationCharacteristic._valueset

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is an observable in the context of CPG-on-EBM-on-FHIR"""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)

        return tx_client.code_in_valueset(
            ObservationCharacteristic.load_valueset(), cc.code, cc.system
        )
