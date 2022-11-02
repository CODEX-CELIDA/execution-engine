import os

from .fhir.client import FHIRClient
from .fhir.terminology import FHIRTerminologyClient
from .omop.webapi import WebAPIClient

tx_client = FHIRTerminologyClient(os.environ["FHIR_TERMINOLOGY_SERVER_URL"])
fhir_client = FHIRClient(os.environ["FHIR_BASE_URL"])
webapi = WebAPIClient(os.environ["OMOP_WEBAPI_URL"])
