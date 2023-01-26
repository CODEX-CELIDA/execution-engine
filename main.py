import logging
from datetime import datetime, timedelta

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

e = ExecutionEngine()
logging.getLogger().setLevel(logging.DEBUG)

for recommendation_url in urls:
    print(recommendation_url)
    cdd = e.load_recommendation(base_url + recommendation_url, force_reload=False)

    e.execute(
        cdd, start_datetime=start_datetime, end_datetime=end_datetime, verbose=True
    )
