import logging

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from execution_engine.clients import tx_client
from execution_engine.constants import VS_LABORATORY_OBSERVATIONS
from execution_engine.converter.characteristic.value import AbstractValueCharacteristic
from execution_engine.fhir.terminology import (
    FHIRTerminologyServerException,
    ValueSetEmptyException,
)
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.vocabulary import LOINC


class LaboratoryCharacteristic(AbstractValueCharacteristic):
    """A laboratory characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement
    _concept_value_static = False

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a laboratory observable in the context of CPG-on-EBM-on-FHIR"""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)

        # todo: needs to provided to the server because it is not an official value set
        try:
            codes = tx_client.get_value_set(VS_LABORATORY_OBSERVATIONS)
        except FHIRTerminologyServerException as e:
            logging.warning(
                f"Terminology server returned error for value set {VS_LABORATORY_OBSERVATIONS}, falling back to system check only."
                f"Error: {e}"
            )
            return LOINC.is_system(cc.system)
        except ValueSetEmptyException:
            logging.warning(
                f"Terminology server returned empty value set {VS_LABORATORY_OBSERVATIONS}, falling back to system check only."
            )
            return LOINC.is_system(cc.system)

        return LOINC.is_system(cc.system) and cc.code in codes
