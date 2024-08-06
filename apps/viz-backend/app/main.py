import json
import re
from typing import List

from database import SessionLocal
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Interval, Recommendation, RecommendationRun
from settings import config
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


# Ensure schema name is a valid identifier
def is_valid_identifier(identifier: str) -> bool:
    """
    Check if a string is a valid identifier.
    """
    return re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier) is not None


result_schema = config.omop.result_schema

if not is_valid_identifier(result_schema):
    raise ValueError("Invalid schema name")


def get_db() -> Session:
    """
    Get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/recommendation/list", response_model=List[Recommendation])
def get_recommendations(db: Session = Depends(get_db)) -> List[Recommendation]:
    """
    Get all recommendations.
    """

    result = db.execute(
        text(
            f"""
            SELECT recommendation_id, recommendation_name, recommendation_title, recommendation_url,
                   recommendation_version, recommendation_package_version, create_datetime
            FROM {result_schema}.recommendation
            """  # nosec: result_schema is checked above (is_valid_identifier)
        )
    )
    return result.fetchall()


@app.get("/recommendation/{recommendation_id}/execution_graph")
def get_execution_graph(recommendation_id: int, db: Session = Depends(get_db)) -> dict:
    """
    Get the execution graph for a specific recommendation by ID.
    """
    result = db.execute(
        text(
            f"""
            SELECT recommendation_execution_graph
            FROM {result_schema}.recommendation
            WHERE recommendation_id = :recommendation_id
            """  # nosec: result_schema is checked above (is_valid_identifier)
        ),
        {"recommendation_id": recommendation_id},
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    # Decode the bytes to a string and parse it as JSON
    execution_graph = json.loads(result.recommendation_execution_graph.decode("utf-8"))

    return {"recommendation_execution_graph": execution_graph}


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
