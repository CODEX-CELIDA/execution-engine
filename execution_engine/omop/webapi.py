import logging
from typing import Dict, List, Optional, Union

import requests

from .concepts import Concept


class WebAPIClient:
    """
    OMOP WebAPI Client
    """

    def __init__(self, api_url: str) -> None:
        assert api_url.startswith("https://") or api_url.startswith(
            "http://"
        ), f"Invalid OMOP WebAPI URL: {api_url}"
        self._api_url = api_url

    def _get(self, url: str) -> Union[List, Dict]:
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

    def _post(self, url: str, params: Union[Dict, List]) -> Union[List, Dict]:
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
        ), f"Could not get resource: HTTP status code {r.status_code}\n{r.text}"
        return r.json()

    def get_concept_info(self, concept_id: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary concept info for the given concept ID.
        """
        logging.info(f"Requesting concept info: {concept_id}")
        return Concept.from_json(
            self._post("/vocabulary/lookup/identifiers/", params=[concept_id])[0]
        )

    def get_standard_concept(self, vocabulary: str, code: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        logging.info(f"Requesting standard concept: {vocabulary} #{code}")
        params = {
            "STANDARD_CONCEPT": "S",
            "VOCABULARY_ID": [vocabulary],
            "QUERY": code,
        }
        c = self._post("/vocabulary/search", params)

        if len(c) > 1:
            # try to find the exact match
            c = [x for x in c if x["CONCEPT_CODE"] == code]

        if len(c) == 0:
            # try a workaround for the OMOP WebAPI bug (mm[Hg] can be found using the GET but not POSt endpoint)
            c = self._get(f"/vocabulary/search/{code}/")
            c = [
                x
                for x in c
                if x["STANDARD_CONCEPT"] == "S" and x["VOCABULARY_ID"] == vocabulary
            ]

            if len(c) > 1:
                # try to find the exact match
                c = [x for x in c if x["CONCEPT_CODE"] == code]

        if len(c) == 0:
            raise Exception(f"Could not find standard concept for {vocabulary}:{code}")
        elif len(c) > 1:
            raise Exception(f"Found multiple standard concepts for {vocabulary}:{code}")

        return Concept.from_json(c[0])
