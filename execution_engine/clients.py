from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir.terminology import FHIRTerminologyClient
from execution_engine.omop.sqlclient import OMOPSQLClient
from execution_engine.settings import get_config

tx_client = FHIRTerminologyClient(get_config().fhir_terminology_server_url)
fhir_client = FHIRClient(get_config().fhir_base_url)
omopdb = OMOPSQLClient(
    **get_config().omop.dict(by_alias=True), timezone=get_config().timezone
)
