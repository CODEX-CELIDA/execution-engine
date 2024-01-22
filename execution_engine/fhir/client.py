import logging

import requests
from fhir.resources import FHIRAbstractModel, construct_fhir_element


class FHIRClient:
    """
    FHIR Client with very basic functionality.

    Used to retrieve resources from a FHIR server by their canonical URL.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def fetch_resource(
        self, element_type: str, canonical_url: str, version: str
    ) -> FHIRAbstractModel:
        """
        Get a resource from the FHIR server by its canonical URL.
        """
        logging.info(f"Requesting resource: {canonical_url}")

        try:
            r = requests.get(
                f"{self.base_url}/{element_type}?url={canonical_url}&version={version}",
                timeout=10,
            )
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Could not connect to FHIR server at {self.base_url}"
            )
        assert (
            r.status_code == 200
        ), f"Could not get resource: HTTP status code {r.status_code}"
        return construct_fhir_element(element_type, r.json())
