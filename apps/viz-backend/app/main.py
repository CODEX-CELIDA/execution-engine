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


@app.get("/execution_runs", response_model=List[RecommendationRun])
def get_execution_runs(db: Session = Depends(get_db)) -> dict:
    """
    Get all recommendation runs.
    """
    result = db.execute(
        text(
            """
    SELECT run_id, observation_start_datetime, observation_end_datetime, run_datetime
    FROM execution_run
    """
        )
    )
    return result.fetchall()


@app.get("/intervals/{run_id}", response_model=List[Interval])
def get_intervals(run_id: int, db: Session = Depends(get_db)) -> dict:
    """
    Get all intervals for a given recommendation run.
    """
    result = db.execute(
        text(
            """
    SELECT *
    FROM interval_result
    WHERE run_id = :run_id
    """
        ),
        {"run_id": run_id},
    )
    return result.fetchall()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec (all interfaces is fine)
