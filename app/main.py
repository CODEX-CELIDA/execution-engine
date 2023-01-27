import sys

sys.path.append("..")
import logging

import pendulum
from fastapi import FastAPI

from execution_engine import ExecutionEngine

e = ExecutionEngine(verbose=True)
recommendations = {}

app = FastAPI()


@app.get("/")
async def root() -> dict:
    """Server greeting"""
    return {"message": "CODEX-CELIDA Execution Engine"}


@app.on_event("startup")
async def startup_event() -> None:
    """
    Load all recommendations
    Returns: None
    """
    base_url = (
        "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
    )

    urls = [
        "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation",
        "sepsis/recommendation/ventilation-plan-ards-tidal-volume",
        "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume",
        "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-ards-peep",
        "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation",
        "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation",
        "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards",
    ]

    logging.getLogger().setLevel(logging.DEBUG)

    for recommendation_url in urls:
        url = base_url + recommendation_url
        logging.info(f"Loading {url}")
        recommendations[url] = e.load_recommendation(url, force_reload=False)


@app.get("/patients/list")
async def patient_list(
    recommendation_url: str, start_datetime: str, end_datetime: str
) -> dict:
    """
    Get list of patients for a specific recommendation.
    """

    assert recommendation_url in recommendations
    cdd = recommendations[recommendation_url]

    run_id = e.execute(
        cdd,
        start_datetime=pendulum.parse(start_datetime),
        end_datetime=pendulum.parse(end_datetime),
        select_patient_data=False,
    )

    res = e.fetch_patients(run_id)[
        ["cohort_category", "criterion_name", "person_id"]
    ].to_dict(orient="list")

    return {"run_id": run_id, "data": res}


@app.get("/patients/data")
async def patient_data(run_id: int, person_id: int, criterion_name: str) -> dict:
    """
    Get individual patient data.
    """

    run = e.fetch_run(run_id)
    # print(run)
    # return run.iloc[0].to_dict()

    start_datetime = run["observation_start_datetime"]
    end_datetime = run["observation_end_datetime"]
    recommendation_url = run["recommendation_url"]

    assert recommendation_url in recommendations

    cdd = recommendations[recommendation_url]

    data = e.fetch_patient_data(
        person_id=person_id,
        criterion_name=criterion_name,
        cdd=cdd,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    return {"res": data}  # .to_dict(orient="list")
