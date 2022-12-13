import logging

from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

from ...clients import tx_client
from ...constants import VS_LABORATORY_OBSERVATIONS
from ...fhir.terminology import ValueSetEmptyException
from ...fhir.util import get_coding
from ...omop.criterion.measurement import Measurement
from ...omop.vocabulary import LOINC
from .value import AbstractValueCharacteristic


class LaboratoryCharacteristic(AbstractValueCharacteristic):
    """A laboratory characteristic in the context of CPG-on-EBM-on-FHIR."""

    _criterion_class = Measurement

    @staticmethod
    def valid(
        char_definition: EvidenceVariableCharacteristic,
    ) -> bool:
        """Checks if characteristic is a laboratory observable in the context of CPG-on-EBM-on-FHIR"""
        cc = get_coding(char_definition.definitionByTypeAndValue.type)

        # todo: needs to provided to the server because it is not an official value set
        try:
            codes = tx_client.get_value_set(VS_LABORATORY_OBSERVATIONS)
        except ValueSetEmptyException:
            logging.warning(
                f"Terminology server returned empty value set {VS_LABORATORY_OBSERVATIONS}, falling back to system check only."
            )
            return LOINC.is_system(cc.system)

        return LOINC.is_system(cc.system) and cc.code in codes
