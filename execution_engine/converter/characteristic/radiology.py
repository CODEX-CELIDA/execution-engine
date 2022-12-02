from ...constants import SCT_RADIOLOGIC_FINDING
from ...omop.criterion.condition_occurrence import ConditionOccurrence
from ...omop.vocabulary import SNOMEDCT
from .codeable_concept import AbstractCodeableConceptCharacteristic


class RadiologyCharacteristic(AbstractCodeableConceptCharacteristic):
    """A radiology characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_RADIOLOGIC_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
