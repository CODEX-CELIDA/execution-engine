import sys
from typing import Any

import pendulum
from app.dependencies import get_execution_engine, get_recommendations

from execution_engine.execution_engine import ExecutionEngine

sys.path.append("..")
from fastapi import APIRouter, Depends, HTTPException

from execution_engine.omop.cohort import Recommendation

router = APIRouter()


@router.get("/patient/list")
async def patient_list(
    recommendation_url: str,
    start_datetime: str,
    end_datetime: str,
    recommendations: dict = Depends(get_recommendations),
    e: ExecutionEngine = Depends(get_execution_engine),
) -> dict:
    """
    Get list of patients for a specific recommendation.
    """
    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    recommendation = recommendations[recommendation_url]["recommendation"]

    run_id = e.execute(
        recommendation,
        start_datetime=pendulum.parse(start_datetime),
        end_datetime=pendulum.parse(end_datetime),
    )

    res = e.fetch_patients(run_id)[
        ["cohort_category", "criterion_name", "person_id"]
    ].to_dict(orient="list")

    return {"run_id": run_id, "data": res}


@router.get("/patient/data")
async def patient_data(
    run_id: int,
    person_id: int,
    criterion_name: str,
    recommendations: dict = Depends(get_recommendations),
    e: ExecutionEngine = Depends(get_execution_engine),
) -> dict[str, Any]:
    """
    Get individual patient data.
    """
    run = e.fetch_run(run_id)

    start_datetime = run["observation_start_datetime"]
    end_datetime = run["observation_end_datetime"]
    recommendation_url = run["recommendation_url"]

    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    recommendation: Recommendation = recommendations[recommendation_url][
        "recommendation"
    ]

    try:
        data = e.fetch_patient_data(
            person_id=person_id,
            criterion_name=criterion_name,
            recommendation=recommendation,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return data.to_dict(orient="list")
