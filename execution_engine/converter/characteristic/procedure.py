from execution_engine.constants import SCT_PROCEDURE
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT


class ProcedureCharacteristic(AbstractCodeableConceptCharacteristic):
    """A procedure characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_PROCEDURE
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ProcedureOccurrence
    _concept_value_static = False
