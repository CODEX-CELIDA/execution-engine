import os

from .fhir.client import FHIRClient
from .fhir.terminology import FHIRTerminologyClient
from .omop.sql import OMOPSQLClient

tx_client = FHIRTerminologyClient(os.environ["FHIR_TERMINOLOGY_SERVER_URL"])
fhir_client = FHIRClient(os.environ["FHIR_BASE_URL"])
omopdb = OMOPSQLClient(
    user=os.environ["OMOP_DB_USER"],
    password=os.environ["OMOP_DB_PASSWORD"],
    host=os.environ["OMOP_DB_HOST"],
    port=int(os.environ["OMOP_DB_PORT"]),
    database=os.environ["OMOP_DB_NAME"],
    schema=os.environ["OMOP_DB_SCHEMA"],
)
