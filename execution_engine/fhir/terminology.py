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
        except requests.ConnectionError:
            raise FHIRTerminologyServerException(
                f"Error getting value set {url}: Connection error"
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

    def code_in_valueset(
        self, valueset_definition: dict, code: str, system: str
    ) -> bool:
        """
        Validate if a code is in a ValueSet using the FHIR $validate-code operation with an embedded ValueSet definition.

        :param valueset_definition: JSON object representing the ValueSet
        :param code: The concept code to validate
        :param system: The coding system the code belongs to
        :return: True if the code is valid within the ValueSet, False otherwise
        """
        # Prepare the request data
        data = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "valueSet",
                    "resource": valueset_definition,  # Embedding the ValueSet resource
                },
                {"name": "code", "valueCode": code},
                {"name": "system", "valueUri": system},
            ],
        }
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        }

        response = requests.post(
            f"{self.server_url}/r5/ValueSet/$validate-code",
            json=data,
            headers=headers,
            timeout=30,
        )

        json_response = response.json()

        if response.status_code == 200:
            # Look for a parameter named 'result' and check if its value is True
            for param in json_response.get("parameter", []):
                if param.get("name") == "result":
                    return param.get("valueBoolean", False)
            return False
        else:
            print(f"Error validating code: HTTP Status {response.status_code}")
            return False
