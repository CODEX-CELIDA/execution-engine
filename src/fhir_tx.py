from typing import List
import requests
import requests_cache

# After installation, all requests functions and Session methods will be cached
requests_cache.install_cache("example_cache", backend="sqlite")


def get_descendents(system: str, code: str) -> List[str]:
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
                                    {"property": "concept", "op": "is-a", "value": code}
                                ],
                            }
                        ]
                    },
                },
            }
        ],
    }
    r = requests.post(
        "http://tx.fhir.org/r4/ValueSet/$expand",
        json=parameters,
        headers={"ACCEPT": "application/fhir+json"},
    )
    return [c["code"] for c in r.json()["expansion"]["contains"]]


def get_value_set(url: str) -> List[str]:
    """
    Expand the given value set from an external FHIR terminology server.
    """
    r = requests.get(
        f"https://fhir.simplifier.net/r4/ValueSet/?url={url}",
        headers={"ACCEPT": "application/fhir+json"},
    )
    return [
        c["code"]
        for c in r.json()["entry"][0]["resource"]["compose"]["include"][0]["concept"]
    ]
