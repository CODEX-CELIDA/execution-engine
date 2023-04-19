import unittest
from unittest.mock import MagicMock

import pytest

from execution_engine.fhir.terminology import (
    FHIRTerminologyClient,
    FHIRTerminologyServerException,
    ValueSetEmptyException,
)


class TestTerminology:
    @pytest.fixture
    def client(self):
        return FHIRTerminologyClient("https://example.com/fhir")

    def test_get_descendents(self, client):
        client.server_url = "https://example.com/fhir"
        system = "http://snomed.info/sct"
        code = "123456789"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "expansion": {
                "contains": [
                    {"code": "234567890"},
                    {"code": "345678901"},
                ]
            }
        }
        with unittest.mock.patch("requests.post", return_value=mock_response):
            result = client.get_descendents(system, code)
            assert result == ["234567890", "345678901"]

    def test_get_value_set(self, client):
        client.server_url = "https://example.com/fhir"
        url = "http://example.com/ValueSet/example"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": {
                        "compose": {
                            "include": [
                                {
                                    "concept": [
                                        {"code": "234567890"},
                                        {"code": "345678901"},
                                    ]
                                }
                            ]
                        }
                    }
                }
            ],
        }
        with unittest.mock.patch("requests.get", return_value=mock_response):
            result = client.get_value_set(url)
            assert result == ["234567890", "345678901"]

    def test_get_value_set_empty(self, client):
        client.server_url = "https://example.com/fhir"
        url = "http://example.com/ValueSet/empty"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "Bundle",
            "type": "searchset",
        }
        with pytest.raises(ValueSetEmptyException):
            with unittest.mock.patch("requests.get", return_value=mock_response):
                client.get_value_set(url)

    def test_get_value_set_wrong_resource_type(self, client):
        client.server_url = "https://example.com/fhir"
        url = "http://example.com/ValueSet/wrong_resource_type"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "WrongType",
            "type": "searchset",
        }
        with pytest.raises(FHIRTerminologyServerException):
            with unittest.mock.patch("requests.get", return_value=mock_response):
                client.get_value_set(url)

    def test_get_value_set_wrong_bundle_type(self, client):
        client.server_url = "https://example.com/fhir"
        url = "http://example.com/ValueSet/wrong_bundle_type"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "Bundle",
            "type": "wrongset",
        }
        with pytest.raises(FHIRTerminologyServerException):
            with unittest.mock.patch("requests.get", return_value=mock_response):
                client.get_value_set(url)
