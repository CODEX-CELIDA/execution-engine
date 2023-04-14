from execution_engine.constants import SCT_RADIOLOGIC_FINDING
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT


class RadiologyCharacteristic(AbstractCodeableConceptCharacteristic):
    """A radiology characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_RADIOLOGIC_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
    _concept_value_static = False
