import os
from datetime import datetime, timedelta

os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/R4"
os.environ["OMOP_DB_USER"] = "postgres"
os.environ["OMOP_DB_PASSWORD"] = "postgres"  # nosec
os.environ["OMOP_DB_HOST"] = "localhost"
os.environ["OMOP_DB_PORT"] = "5432"
os.environ["OMOP_DB_NAME"] = "ohdsi"
os.environ["OMOP_DB_SCHEMA"] = "cds_cdm"

from execution_engine import ExecutionEngine
from execution_engine.clients import omopdb

base_url = (
    "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/"
)

urls = [
    # "intervention-plan/antithrombotic-prophylaxis-LMWH",
    # "intervention-plan/therapeutic-anticoagulation",
    # "intervention-plan/antithrombotic-prophylaxis-fondaparinux",
    # "intervention-plan/no-antithrombotic-prophylaxis",
    # "intervention-plan/ventilation-plan",
    "intervention-plan/peep-for-ards-fio2-point3",
    # "intervention-plan/abdominal-positioning-ARDS-plan",
]

datetime_start = datetime.today() - timedelta(days=7)
datetime_end = datetime.today()
table_name_output = "recommendation_patients"


e = ExecutionEngine()

for recommendation_url in urls:
    print(recommendation_url)
    cd = e.process_recommendation(base_url + recommendation_url)
    for statement in cd.process(table_name_output, datetime_start, datetime_end):
        print(statement)
        omopdb.execute(statement)
