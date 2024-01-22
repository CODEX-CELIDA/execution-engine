import unittest
from unittest.mock import MagicMock

import pytest

from execution_engine.fhir.client import FHIRClient


class TestFHIRClient:
    @pytest.fixture
    def client(self):
        return FHIRClient("https://example.com/fhir")

    def test_fetch_resource(self, client):
        client.base_url = "https://example.com/fhir"
        element_type = "ValueSet"
        canonical_url = "http://example.com/ValueSet/example"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "ValueSet",
            "url": canonical_url,
            "status": "active",
        }
        with unittest.mock.patch("requests.get", return_value=mock_response):
            result = client.fetch_resource(
                element_type, canonical_url, version="latest"
            )
            assert result.url == canonical_url
            assert result.status == "active"

    def test_fetch_resource_connection_refused(self, client):
        client.base_url = "https://badurl.com/fhir"
        element_type = "ValueSet"
        canonical_url = "http://example.com/ValueSet/example"

        with unittest.mock.patch("requests.get", side_effect=ConnectionRefusedError):
            with pytest.raises(ConnectionRefusedError):
                client.fetch_resource(element_type, canonical_url, version="latest")

    def test_fetch_resource_bad_status_code(self, client):
        client.base_url = "https://example.com/fhir"
        element_type = "ValueSet"
        canonical_url = "http://example.com/ValueSet/example"

        mock_response = MagicMock()
        mock_response.status_code = 404
        with unittest.mock.patch("requests.get", return_value=mock_response):
            with pytest.raises(AssertionError):
                client.fetch_resource(element_type, canonical_url, version="latest")
