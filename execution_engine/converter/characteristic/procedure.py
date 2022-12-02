from ...constants import SCT_PROCEDURE
from ...omop.criterion.procedure_occurrence import ProcedureOccurrence
from ...omop.vocabulary import SNOMEDCT
from .codeable_concept import AbstractCodeableConceptCharacteristic


class ProcedureCharacteristic(AbstractCodeableConceptCharacteristic):
    """A procedure characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_PROCEDURE
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ProcedureOccurrence
