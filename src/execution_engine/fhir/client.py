from pathlib import Path
from typing import Dict

import requests
from fhir.resources import construct_fhir_element
from fhir.resources.resource import Resource


class FHIRClient:
    """
    FHIR Client with very basic functionality.

    Used to retrieve resources from a FHIR server by their canonical URL.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_resource(self, element_type: str, canonical_url: str) -> Resource:
        """
        Get a resource from the FHIR server by its canonical URL.
        """
        r = requests.get(f"{self.base_url}/{element_type}?url={canonical_url}")
        assert (
            r.status_code == 200
        ), f"Could not get resource: HTTP status code {r.status_code}"
        return construct_fhir_element(element_type, r.json())
