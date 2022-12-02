from ...constants import SCT_CLINICAL_FINDING
from ...omop.criterion.condition_occurrence import ConditionOccurrence
from ...omop.vocabulary import SNOMEDCT
from .codeable_concept import AbstractCodeableConceptCharacteristic


class ConditionCharacteristic(AbstractCodeableConceptCharacteristic):
    """A condition characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_CLINICAL_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
