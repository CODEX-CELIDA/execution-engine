from execution_engine.constants import LOINC_EPISODE_OF_CARE_TYPE
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.criterion.visit_occurrence import VisitOccurrence
from execution_engine.omop.vocabulary import LOINC


class EpisodeOfCareCharacteristic(AbstractCodeableConceptCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = LOINC_EPISODE_OF_CARE_TYPE
    _concept_vocabulary = LOINC
    _criterion_class = VisitOccurrence
    _concept_value_static = False
