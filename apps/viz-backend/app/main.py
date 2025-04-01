import json
import re
from typing import List, cast

from database import SessionLocal
from fastapi import Depends, FastAPI, HTTPException, Query
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
data_schema = config.omop.data_schema

if not is_valid_identifier(result_schema):
    raise ValueError("Invalid schema name")

if not is_valid_identifier(data_schema):
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


@app.get("/execution_run/list", response_model=List[RecommendationRun])
def get_execution_runs(db: Session = Depends(get_db)) -> dict:
    """
    Get all recommendation runs.
    """
    result = db.execute(
        text(
            f"""
    SELECT run_id, observation_start_datetime, observation_end_datetime, run_datetime, r.recommendation_name
    FROM {result_schema}.execution_run
    INNER JOIN {result_schema}.recommendation r ON r.recommendation_id = execution_run.recommendation_id
    """  # nosec: result_schema is checked above (is_valid_identifier)
        )
    )
    return result.fetchall()


@app.get("/execution_run/{run_id}/person_ids", response_model=List[int])
def get_patients(run_id: int, db: Session = Depends(get_db)) -> list[int]:
    """
    Get all patients for a given recommendation run.
    """
    result = db.execute(
        text(
            f"""
    SELECT DISTINCT person_id
    FROM {result_schema}.result_interval
    WHERE run_id = :run_id
    """  # nosec: result_schema is checked above (is_valid_identifier)
        ),
        {"run_id": run_id},
    )
    return [r[0] for r in result.fetchall()]


@app.get("/intervals/{run_id}", response_model=List[Interval])
def get_intervals(
    run_id: int,
    person_id: int | None = None,
    person_source_value: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get all intervals for a given recommendation run.
    """
    if person_id is None and person_source_value is None:
        raise HTTPException(
            status_code=400,
            detail="Either person_id or person_source_value must be provided",
        )
    if person_id is not None and person_source_value is not None:
        raise HTTPException(
            status_code=400,
            detail="Only one of person_id or person_source_value can be provided",
        )

    params: dict[str, int | str] = {"run_id": run_id}

    if person_id is not None:
        query = f"""
            SELECT *
            FROM {result_schema}.interval_result
            WHERE run_id = :run_id
            AND person_id = :person_id
            """  # nosec: result_schema is checked above (is_valid_identifier)
        params["person_id"] = person_id
    elif person_source_value is not None:
        query = f"""
            SELECT ir.*
            FROM {result_schema}.interval_result ir
            INNER JOIN person p ON ir.person_id = p.person_id
            WHERE ir.run_id = :run_id
            AND p.person_source_value = :person_source_value
            """  # nosec: result_schema is checked above (is_valid_identifier)
        params["person_source_value"] = str(person_source_value)

    result = db.execute(text(query), params)
    return result.fetchall()


#########################################
# Person Data


tables = {
    "condition_occurrence": {
        "concept_id": "condition_concept_id",
        "columns": [
            "condition_start_datetime",
            "condition_end_datetime",
        ],
        "sort_keys": ["condition_start_datetime"],
    },
    "measurement": {
        "concept_id": "measurement_concept_id",
        "columns": [
            "measurement_datetime",
            "value_as_number",
            "value_as_concept_id",
            "unit_concept_id",
            "measurement_source_value",
            "unit_source_value",
        ],
        "sort_keys": ["measurement_datetime"],
    },
    "drug_exposure": {
        "concept_id": "drug_concept_id",
        "columns": [
            "drug_exposure_start_datetime",
            "drug_exposure_end_datetime",
            "quantity",
            "route_concept_id",
            "drug_source_value",
            "drug_source_concept_id",
        ],
        "sort_keys": ["drug_exposure_start_datetime", "drug_exposure_end_datetime"],
    },
    "procedure_occurrence": {
        "concept_id": "procedure_concept_id",
        "columns": [
            "procedure_datetime",
            "procedure_end_datetime",
            "procedure_source_value",
            "procedure_source_concept_id",
        ],
        "sort_keys": ["procedure_datetime"],
    },
    "observation": {
        "concept_id": "observation_concept_id",
        "columns": [
            "observation_datetime",
            "value_as_number",
            "value_as_concept_id",
            "observation_source_value",
            "observation_source_concept_id",
        ],
        "sort_keys": ["observation_datetime"],
    },
    "visit_occurrence": {
        "concept_id": "visit_concept_id",
        "columns": [
            "visit_start_datetime",
            "visit_end_datetime",
            "visit_source_value",
            "visit_source_concept_id",
        ],
        "sort_keys": ["visit_start_datetime"],
    },
    "visit_detail": {
        "concept_id": "visit_detail_concept_id",
        "columns": [
            "visit_detail_start_datetime",
            "visit_detail_end_datetime",
            "visit_detail_source_value",
            "visit_detail_source_concept_id",
        ],
        "sort_keys": ["visit_detail_start_datetime"],
    },
}


@app.get("/resolve_person_id/{person_source_value}")
async def resolve_person_id(
    person_source_value: str, db: Session = Depends(get_db)
) -> dict:
    """
    Get person_id from person_source_value
    """

    result = db.execute(
        text(
            f"SELECT person_id FROM {data_schema}.person WHERE person_source_value = :person_source_value"  # nosec: result_schema is checked above (is_valid_identifier)
        ),
        {"person_source_value": person_source_value},
    )
    res = result.fetchone()

    if res:
        return {"person_id": res[0]}
    else:
        raise HTTPException(status_code=404, detail="Person not found")


@app.get("/patient/{person_id}")
async def get_patient_data(person_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Get all data for a given person id
    """
    data = {}

    for table_name in tables:

        table = tables[table_name]
        select_additional = ""
        join_additional = ""

        if "value_as_concept_id" in table["columns"]:
            select_additional = ", c_value.concept_name as value_concept_name"
            join_additional = f"LEFT JOIN {data_schema}.concept c_value ON t.value_as_concept_id = c_value.concept_id"

        query = text(
            f"""
            SELECT t.*, c.concept_name, c.concept_id {select_additional}
            FROM {data_schema}.{table_name} as t
            JOIN {data_schema}.concept c ON t.{table['concept_id']} = c.concept_id
            {join_additional}
            WHERE person_id = :person_id"""  # nosec: result_schema and data_schema are checked above (is_valid_identifier)
        )

        result = db.execute(query, {"person_id": person_id})
        columns = [desc.name for desc in result.cursor.description]
        rows = result.fetchall()
        data[table_name] = [dict(zip(columns, row)) for row in rows]

    return data


@app.get("/concepts/{person_id}/{table_name}")
async def get_concepts(
    person_id: str,
    table_name: str,
    concept_id: int = Query(...),
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Get all concepts for a given person and table.
    """

    table = tables[table_name]
    columns: list[str] = [cast(str, table["concept_id"])]
    concept_joins: list[str] = []

    for column in table["columns"]:

        if column.endswith("_concept_id"):
            n_joins = len(concept_joins)
            concept_table_name = f"concept_{n_joins}"
            columns.append(f"{concept_table_name}.concept_name as {column[:-3]}_name")
            concept_joins.append(
                f"LEFT JOIN {data_schema}.concept {concept_table_name} ON t.{column} = {concept_table_name}.concept_id"
            )
        else:
            columns.append(f"t.{column}")

    columns_str = ", ".join(columns)
    concept_joins_str = " ".join(concept_joins)

    order_by_cols = ", ".join([f"t.{c}" for c in table["sort_keys"]])

    query = text(
        f"""
        SELECT {columns_str}
        FROM {data_schema}.{table_name} t
        {concept_joins_str}
        WHERE person_id = :person_id AND {table["concept_id"]} = :concept_id
        ORDER BY {order_by_cols}
        """  # nosec: result_schema is checked above (is_valid_identifier) or fields are hardcoded
    )

    result = db.execute(query, {"person_id": person_id, "concept_id": concept_id})
    columns = [desc.name for desc in result.cursor.description]
    rows = result.fetchall()
    data = [dict(zip(columns, row)) for row in rows]

    return data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec (all interfaces is fine)
