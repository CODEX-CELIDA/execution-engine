from typing import List

from database import SessionLocal
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Interval, RecommendationRun
from sqlalchemy import text
from sqlalchemy.orm import Session

app = FastAPI()

# Set up CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_db() -> Session:
    """
    Get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/recommendation_runs", response_model=List[RecommendationRun])
def get_recommendation_runs(db: Session = Depends(get_db)) -> dict:
    """
    Get all recommendation runs.
    """
    result = db.execute(
        text(
            """
    SELECT recommendation_run_id, observation_start_datetime, observation_end_datetime, run_datetime
    FROM recommendation_run
    """
        )
    )
    return result.fetchall()


@app.get("/intervals/{recommendation_run_id}", response_model=List[Interval])
def get_intervals(recommendation_run_id: int, db: Session = Depends(get_db)) -> dict:
    """
    Get all intervals for a given recommendation run.
    """
    result = db.execute(
        text(
            """
    SELECT rri.person_id, rri.pi_pair_id, rri.criterion_id, pip.pi_pair_name,
           rc.criterion_name, rri.interval_type, rri.interval_start, rri.interval_end,
           rr.observation_start_datetime, rr.observation_end_datetime, rri.cohort_category
    FROM recommendation_result_interval rri
    LEFT JOIN population_intervention_pair pip ON rri.pi_pair_id = pip.pi_pair_id
    LEFT JOIN recommendation_criterion rc ON rri.criterion_id = rc.criterion_id
    JOIN recommendation_run rr ON rri.recommendation_run_id = rr.recommendation_run_id
    WHERE rri.recommendation_run_id = :recommendation_run_id
    """
        ),
        {"recommendation_run_id": recommendation_run_id},
    )
    return result.fetchall()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec (all interfaces is fine)
