from enum import Enum
from typing import Dict

import requests

from .. import LOINC, SNOMEDCT


class Vocabulary(Enum):
    """
    OMOP Vocabulary IDs
    """

    SNOMEDCT = "SNOMED"
    LOINC = "LOINC"

    def from_url(url: str) -> "Vocabulary":
        """
        Get the vocabulary ID from the system URL.
        """
        if url == SNOMEDCT:
            return Vocabulary.SNOMEDCT
        elif url == LOINC:
            return Vocabulary.LOINC
        else:
            raise ValueError(f"Unknown vocabulary: {url}")
