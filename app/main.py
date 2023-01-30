import sys
from typing import TypedDict

from execution_engine.omop.cohort_definition import CohortDefinitionCombination
from execution_engine.omop.criterion.concept import ConceptCriterion

sys.path.append("..")
import logging

import pendulum
from fastapi import FastAPI, HTTPException

from execution_engine import ExecutionEngine


class Recommendation(TypedDict):
    """
    Recommendation for execution engine (for type hinting).
    """

    recommendation_name: str
    recommendation_title: str
    cohort_definition: CohortDefinitionCombination


recommendations: dict[str, Recommendation] = {}
e = ExecutionEngine(verbose=False)
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

    for recommendation_url in urls:
        url = base_url + recommendation_url
        logging.info(f"Loading {url}")
        rec = e.load_recommendation(url, force_reload=False)
        recommendations[url] = {
            "recommendation_name": rec.name,
            "recommendation_title": rec.title,
            "cohort_definition": rec,
        }


@app.get("/recommendation/list")
async def recommendation_list() -> list[dict[str, str]]:
    """
    Get available recommendations by URL
    """
    return [
        {
            "recommendation_name": recommendations[url]["recommendation_name"],
            "recommendation_title": recommendations[url]["recommendation_title"],
            "recommendation_url": url,
        }
        for url in recommendations
    ]


@app.get("/recommendation/criteria")
async def recommendation_criteria(recommendation_url: str) -> dict:
    """
    Get criteria names by recommendation URL
    """

    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    cdd: CohortDefinitionCombination = recommendations[recommendation_url][
        "cohort_definition"
    ]
    criteria = cdd.criteria()
    data = []

    for c in criteria:
        data.append(
            {
                "unique_name": c.unique_name(),
                "concept_name": c.concept.concept_name
                if isinstance(c, ConceptCriterion)
                else None,
            }
        )

    return {"criterion": data}


@app.get("/patient/list")
async def patient_list(
    recommendation_url: str, start_datetime: str, end_datetime: str
) -> dict:
    """
    Get list of patients for a specific recommendation.
    """

    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    cdd = recommendations[recommendation_url]["cohort_definition"]

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


@app.get("/patient/data")
async def patient_data(run_id: int, person_id: int, criterion_name: str) -> dict:
    """
    Get individual patient data.
    """

    run = e.fetch_run(run_id)

    start_datetime = run["observation_start_datetime"]
    end_datetime = run["observation_end_datetime"]
    recommendation_url = run["recommendation_url"]

    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    cdd: CohortDefinitionCombination = recommendations[recommendation_url][
        "cohort_definition"
    ]

    try:
        data = e.fetch_patient_data(
            person_id=person_id,
            criterion_name=criterion_name,
            cdd=cdd,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {"res": data}  # .to_dict(orient="list")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(  # nosec (binding to all interfaces is desired)
        app, host="0.0.0.0", port=8001
    )
