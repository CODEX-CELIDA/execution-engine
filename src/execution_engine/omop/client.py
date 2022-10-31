import json
import logging
import os
from typing import Dict, List, Optional

import requests
import requests_cache

from .concepts import Concept
from .vocabulary import Vocabulary


class WebAPIClient:
    """
    OMOP WebAPI Client
    """

    def __init__(self, api_url: str) -> None:
        assert api_url.startswith("https://") or api_url.startswith(
            "http://"
        ), f"Invalid OMOP WebAPI URL: {api_url}"
        self._api_url = api_url

    def _get(self, url: str) -> Dict:
        """
        GET request to OMOP WebAPI
        """
        try:
            r = requests.get(self._api_url + url)
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to OMOP WebAPI at {self._api_url}"
            ) from e
        assert (
            r.status_code == 200
        ), f"Could not get resource: HTTP status code {r.status_code}"
        return r.json()

    def _post(self, url: str, params: Dict) -> Dict:
        """
        POST request to OMOP WebAPI
        """
        try:
            r = requests.post(self._api_url + url, json=params)
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to OMOP WebAPI at {self._api_url}"
            ) from e
        assert (
            r.status_code == 200
        ), f"Could not get resource: HTTP status code {r.status_code}"
        return r.json()

    def get_concept_info(self, concept_id: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary concept info for the given concept ID.
        """
        logging.info(f"Requesting concept info: {concept_id}")
        return Concept.from_json(self._get(f"/vocabulary/{concept_id}"))

    def get_standard_concept(self, vocabulary: Vocabulary, code: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        logging.info(f"Requesting standard concept: {vocabulary.value} {code}")
        params = {
            "STANDARD_CONCEPT": "S",
            "VOCABULARY_ID": [vocabulary.value],
            "QUERY": code,
        }
        c = self._post("/vocabulary/search", params)

        if len(c) == 0:
            raise Exception(
                f"Could not find standard concept for {vocabulary.name}:{code}"
            )
        elif len(c) > 1:
            raise Exception(
                f"Found multiple standard concepts for {vocabulary.name}:{code}"
            )

        return Concept.from_json(c[0])

    def create_cohort(
        self,
        name: str,
        description: str,
        definition: Dict,
        tags: Optional[List[str]] = None,
    ) -> Dict:
        """Create a cohort defition in the OMOP CDM"""
        params = {
            "name": name,
            "description": description,
            # "hasWriteAccess": True,
            "tags": tags if tags is not None else [],
            "expressionType": "SIMPLE_EXPRESSION",
            "expression": definition,
        }

        return self._post("/cohortdefinition", params)
