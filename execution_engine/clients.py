from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir.terminology import FHIRTerminologyClient
from execution_engine.omop.sql import OMOPSQLClient
from execution_engine.settings import config

tx_client = FHIRTerminologyClient(config.fhir_terminology_server_url)
fhir_client = FHIRClient(config.fhir_base_url)
omopdb = OMOPSQLClient(**config.omop.dict(by_alias=True))
