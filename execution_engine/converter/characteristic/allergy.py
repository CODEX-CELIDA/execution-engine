from ...constants import SCT_ALLERGIC_DISPOSITION
from ...omop.criterion.condition_occurrence import ConditionOccurrence
from ...omop.vocabulary import SNOMEDCT
from .codeable_concept import AbstractCodeableConceptCharacteristic


class AllergyCharacteristic(AbstractCodeableConceptCharacteristic):
    """An allergy characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_ALLERGIC_DISPOSITION
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
    _concept_value_static = True
