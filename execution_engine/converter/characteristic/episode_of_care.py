from ...constants import LOINC_EPISODE_OF_CARE_TYPE
from ...omop.criterion.visit_occurrence import VisitOccurrence
from ...omop.vocabulary import LOINC
from .codeable_concept import AbstractCodeableConceptCharacteristic


class EpisodeOfCareCharacteristic(AbstractCodeableConceptCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = LOINC_EPISODE_OF_CARE_TYPE
    _concept_vocabulary = LOINC
    _criterion_class = VisitOccurrence
