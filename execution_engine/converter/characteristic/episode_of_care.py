from typing import Type, cast

from execution_engine.constants import LOINC_EPISODE_OF_CARE_TYPE
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.criterion.visit_detail import VisitDetail
from execution_engine.omop.criterion.visit_occurrence import VisitOccurrence
from execution_engine.omop.vocabulary import LOINC
from execution_engine.settings import config


class EpisodeOfCareCharacteristic(AbstractCodeableConceptCharacteristic):
    """An episode of care characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = LOINC_EPISODE_OF_CARE_TYPE
    _concept_vocabulary = LOINC
    _criterion_class = cast(
        Type[ConceptCriterion],
        VisitOccurrence
        if not config.celida_ee_episode_of_care_visit_detail
        else VisitDetail,
    )
    _concept_value_static = False
