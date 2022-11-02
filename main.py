import os

os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["OMOP_WEBAPI_URL"] = "http://192.168.200.128:9876/WebAPI"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/R4"

from execution_engine.execution_engine import ExecutionEngine

# recommendation_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/intervention-plan/antithrombotic-prophylaxis-LMWH"
recommendation_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/intervention-plan/therapeutic-anticoagulation"

e = ExecutionEngine()

cd = e.process_recommendation(recommendation_url)
