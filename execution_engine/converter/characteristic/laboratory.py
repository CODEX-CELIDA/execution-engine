from fhir.resources.evidencevariable import EvidenceVariableCharacteristic

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
        # TODO: Don't just use all LOINC codes, but restrict to subset of important ones (or actually used ones)
        return LOINC.is_system(cc.system)
