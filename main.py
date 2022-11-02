import os

os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["OMOP_WEBAPI_URL"] = "http://192.168.200.128:9876/WebAPI"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/R4"

from execution_engine import ExecutionEngine

base_url = (
    "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/"
)

urls = [
    "intervention-plan/antithrombotic-prophylaxis-LMWH",
    "intervention-plan/therapeutic-anticoagulation",
    "intervention-plan/antithrombotic-prophylaxis-fondaparinux",
    "intervention-plan/no-antithrombotic-prophylaxis",
    "intervention-plan/ventilation-plan",
    "intervention-plan/peep-for-ards-fio2-point3",
    "intervention-plan/abdominal-positioning-ARDS-plan",
]

e = ExecutionEngine()

for recommendation_url in urls:
    print(recommendation_url)
    cd = e.process_recommendation(base_url + recommendation_url)
