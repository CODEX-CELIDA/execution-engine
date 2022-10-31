import json
import logging
import os
from typing import Dict

import requests
import requests_cache

from .vocabulary import Vocabulary

# API_URL = os.getenv("OMOP_WEBAPI_URL")


class WebAPIClient:
    """
    OMOP WebAPI Client
    """

    def __init__(self, api_url: str) -> None:
        self._api_url = api_url

    def get_concept_info(self, concept_id: str) -> Dict:
        """
        Get the OMOP Standard Vocabulary concept info for the given concept ID.
        """
        r = requests.get(self._api_url + f"/vocabulary/concept/{concept_id}")
        assert r.status_code == 200
        return r.json()

    def get_standard_concept(self, vocabulary: Vocabulary, code: str) -> int:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """

        params = {
            "STANDARD_CONCEPT": "S",
            "VOCABULARY_ID": [vocabulary.value],
            "QUERY": code,
        }

        r = requests.post(self._api_url + "/vocabulary/search", json=params)
        assert r.status_code == 200
        c = r.json()

        if len(c) == 0:
            raise Exception(
                f"Could not find standard concept for {vocabulary.name}:{code}"
            )
        elif len(c) > 1:
            raise Exception(
                f"Found multiple standard concepts for {vocabulary.name}:{code}"
            )

        return c[0]
