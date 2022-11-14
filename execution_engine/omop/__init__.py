"""
OMOP package to represent OMOP CDM concepts and cohort definitions.
"""
from enum import Enum


class StandardConcepts(Enum):
    """
    Collection of standard concepts in the OMOP CDM.
    """

    VISIT_TYPE_STILL_PATIENT = 32220
