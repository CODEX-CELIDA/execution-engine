from typing import Dict
import requests
from enum import Enum
from . import API_URL


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
        if url == "http://snomed.info/sct":  # TODO replace by constant
            return Vocabulary.SNOMEDCT
        elif url == "http://loinc.org":  # TODO replace by constant
            return Vocabulary.LOINC
        else:
            raise ValueError(f"Unknown vocabulary: {url}")


def get_concept_info(concept_id: str) -> Dict:
    """
    Get the OMOP Standard Vocabulary concept info for the given concept ID.
    """
    r = requests.get(API_URL + f"/vocabulary/concept/{concept_id}")
    assert r.status_code == 200
    return r.json()


def get_standard_concept(vocabulary: Vocabulary, code: str) -> int:
    """
    Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
    """

    params = {
        "STANDARD_CONCEPT": "S",
        "VOCABULARY_ID": [vocabulary.value],
        "QUERY": code,
    }

    r = requests.post(API_URL + "/vocabulary/search", json=params)
    assert r.status_code == 200
    c = r.json()

    if len(c) == 0:
        raise Exception(f"Could not find standard concept for {vocabulary.name}:{code}")
    elif len(c) > 1:
        raise Exception(
            f"Found multiple standard concepts for {vocabulary.name}:{code}"
        )

    return c[0]
