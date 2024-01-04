import requests

# fixme use caching?
# import requests_cache
# requests_cache.install_cache("example_cache", backend="sqlite")


class FHIRTerminologyServerException(Exception):
    """
    Raised when a terminology server returns an error.
    """


class ValueSetEmptyException(FHIRTerminologyServerException):
    """
    Raised when a value set retrieved from the terminology server has no entries.
    """


class FHIRTerminologyClient:
    """
    FHIR Terminology Server Client
    """

    def __init__(self, tx_server_url: str):
        self.server_url = tx_server_url

    def get_descendents(self, system: str, code: str) -> list[str]:
        """
        Get the descendents (concepts) of the given code in the given system from an external FHIR terminology server.
        """
        parameters = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "valueSet",
                    "resource": {
                        "resourceType": "ValueSet",
                        "compose": {
                            "include": [
                                {
                                    "system": system,
                                    "filter": [
                                        {
                                            "property": "concept",
                                            "op": "is-a",
                                            "value": code,
                                        }
                                    ],
                                }
                            ]
                        },
                    },
                }
            ],
        }
        r = requests.post(
            f"{self.server_url}/ValueSet/$expand",
            json=parameters,
            headers={"ACCEPT": "application/fhir+json"},
            timeout=30,
        )
        return [c["code"] for c in r.json()["expansion"]["contains"]]

    def get_value_set(self, url: str) -> list[str]:
        """
        Expand the given value set from an external FHIR terminology server.
        """
        try:
            r = requests.get(
                f"{self.server_url}/ValueSet/?url={url}",
                headers={"ACCEPT": "application/fhir+json"},
                timeout=10,
            )
        except requests.ReadTimeout:
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Request timed out"
            )

        if r.status_code == 500:
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Internal server error"
            )
        elif r.status_code != 200:
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Status Code {r.status_code}\n{r.text}"
            )

        json = r.json()
        if json["resourceType"] != "Bundle":
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Found resource type {json['resourceType']}, expected Bundle"
            )
        if json["type"] != "searchset":
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Found bundle type {json['type']}, expected searchset"
            )

        if "entry" not in json:
            raise ValueSetEmptyException(
                f"Error getting value set {url}: No entries found in bundle"
            )

        return [
            c["code"]
            for c in json["entry"][0]["resource"]["compose"]["include"][0]["concept"]
        ]
