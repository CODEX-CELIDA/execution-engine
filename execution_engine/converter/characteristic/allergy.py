from execution_engine.constants import SCT_ALLERGIC_DISPOSITION
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT


class AllergyCharacteristic(AbstractCodeableConceptCharacteristic):
    """An allergy characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_ALLERGIC_DISPOSITION
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
    _concept_value_static = True
