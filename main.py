import os
from datetime import datetime, timedelta

os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/r4"
os.environ["OMOP_DB_USER"] = "postgres"
os.environ["OMOP_DB_PASSWORD"] = "postgres"  # nosec
os.environ["OMOP_DB_HOST"] = "localhost"
os.environ["OMOP_DB_PORT"] = "5432"
os.environ["OMOP_DB_NAME"] = "ohdsi"
os.environ["OMOP_DB_SCHEMA"] = "cds_cdm"

from execution_engine import ExecutionEngine

base_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"

urls = [
    "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation",
    "sepsis/recommendation/ventilation-plan-ards-tidal-volume",
    "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume",
    "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-ards-peep",
    "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation",
    "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation",
    "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards",
]

start_datetime = datetime.today() - timedelta(days=7)
end_datetime = datetime.today()
table_name_output = "recommendation_patients"


e = ExecutionEngine()


for recommendation_url in urls:
    print(recommendation_url)
    cdd = e.load_recommendation(base_url + recommendation_url, force_reload=False)

    e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)

    break
