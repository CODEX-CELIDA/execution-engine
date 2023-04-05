import logging
from typing import Type

from execution_engine import constants
from execution_engine.constants import SCT_CLINICAL_FINDING
from execution_engine.converter.characteristic.allergy import AllergyCharacteristic
from execution_engine.converter.characteristic.codeable_concept import (
    AbstractCodeableConceptCharacteristic,
)
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT, standard_vocabulary


def is_allergy(concept: Concept) -> bool:
    """Checks if the given concept is an allergy."""

    return standard_vocabulary.related_to(
        constants.OMOP_ALLERGY, concept.concept_id, "Pathology of"
    )


class ConditionCharacteristic(AbstractCodeableConceptCharacteristic):
    """A condition characteristic in the context of CPG-on-EBM-on-FHIR."""

    _concept_code = SCT_CLINICAL_FINDING
    _concept_vocabulary = SNOMEDCT
    _criterion_class = ConditionOccurrence
    _concept_value_static = False

    @classmethod
    def get_class_from_concept(
        cls, concept: Concept
    ) -> Type["AbstractCodeableConceptCharacteristic"]:
        """Gets the class that matches the given concept."""

        if is_allergy(concept):
            logging.warning(
                "ConditionCharacteristic.get_class_from_concept: concept is an allergy but denoted as condition in the FHIR representation."
            )
            return AllergyCharacteristic

        return cls
