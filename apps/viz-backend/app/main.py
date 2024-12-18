import datetime
import json
import re
from typing import List

from database import (
    SessionLocal,
    concept,
    condition_occurrence,
    criterion,
    drug_exposure,
    full_day_coverage,
    interval_result,
    measurement,
    observation,
    partial_day_coverage,
    person,
    procedure_occurrence,
    visit_occurrence,
)
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from models import DayCoverage, Interval, Recommendation, RecommendationRun
from settings import config
from sqlalchemy import and_, func, join, or_, select, text, union
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
    SELECT run_id, observation_start_datetime, observation_end_datetime, run_datetime,
    r.recommendation_name, r.recommendation_title, r.recommendation_url, r.recommendation_description
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
    person_source_value: int | None = None,
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

    # Base query
    base_query = select(interval_result).where(interval_result.c.run_id == run_id)

    # Add filters for person_id or person_source_value
    if person_id is not None:
        base_query = base_query.where(interval_result.c.person_id == person_id)
    elif person_source_value is not None:
        joined_table = join(
            interval_result, person, interval_result.c.person_id == person.c.person_id
        )
        base_query = (
            select(interval_result)
            .select_from(joined_table)
            .where(
                and_(
                    interval_result.c.run_id == run_id,
                    person.c.person_source_value == person_source_value,
                )
            )
        )

    # Execute query
    result = db.execute(base_query)
    rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No intervals found for run_id {run_id} with the specified parameters.",
        )

    return result.fetchall()


@app.get("/criteria/{run_id}/{person_id}", response_model=List[dict])
def get_criteria_for_person(
    person_id: int,
    run_id: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Retrieve all raw data criteria for a person within a specified time range.
    """
    # Step 1: Get all criterion_ids from result_interval where criterion_id is not NULL
    query = (
        select(
            interval_result.c.criterion_id,
            interval_result.c.cohort_category,
            criterion.c.criterion_concept_id,
            concept.c.concept_name,
        )
        .join(criterion, interval_result.c.criterion_id == criterion.c.criterion_id)
        .join(
            concept,
            criterion.c.criterion_concept_id == concept.c.concept_id,
            isouter=True,
        )
        .where(
            and_(
                interval_result.c.person_id == person_id,
                interval_result.c.criterion_id.is_not(None),
                # interval_result.c.interval_start >= start_datetime,
                # interval_result.c.interval_end <= end_datetime,
                interval_result.c.run_id == run_id,
            )
        )
        .distinct()
    )

    result = db.execute(query)
    rows = result.fetchall()

    return [
        {
            "criterion_id": row.criterion_id,
            "concept_id": row.criterion_concept_id,
            "concept_name": row.concept_name,
            "cohort_category": row.cohort_category,
        }
        for row in rows
    ]


### SECOND ENDPOINT: Retrieve raw data for a person and criterion
@app.get("/person/{person_id}/data", response_model=List[dict])
def get_raw_data_for_person(
    person_id: int,
    concept_id: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Retrieve raw data for a person based on a criterion ID within a specified time range.
    """
    # Step 1: Resolve domain_id from the concept table via criterion_id
    domain_query = (
        select(concept.c.domain_id)
        # .join(criterion, criterion.c.concept_id == concept.c.concept_id)
        .where(concept.c.concept_id == concept_id)
    )
    domain_result = db.execute(domain_query).scalar()

    if not domain_result:
        raise HTTPException(
            status_code=404, detail=f"No domain found for concept_id {concept_id}"
        )

    # Step 2: Identify the correct table based on domain_id
    table_mapping = {
        "Measurement": (tables["measurement"], measurement),
        "Visit": (tables["visit_occurrence"], visit_occurrence),
        "Drug": (tables["drug_exposure"], drug_exposure),
        "Observation": (tables["observation"], observation),
        "Condition": (tables["condition_occurrence"], condition_occurrence),
        "Procedure": (tables["procedure_occurrence"], procedure_occurrence),
    }
    target_table = table_mapping.get(domain_result)

    if not target_table:
        raise HTTPException(
            status_code=400, detail=f"No table mapped for domain_id {domain_result}"
        )

    # condition_occurrence": {
    #         "concept_id": "condition_concept_id",
    #         "columns": [
    #             "condition_start_datetime",
    #             "condition_end_datetime",
    #         ],
    #         "sort_keys": ["condition_start_datetime"],
    #     },

    # Step 3: Query the relevant table
    t = target_table[1]
    columns = target_table[0]
    col_concept = columns["concept_id"]

    if t == measurement:
        cols = select(
            t.c.measurement_datetime.label("start_datetime"),
            t.c.value_as_number.label("value"),
            t.c.unit_concept_id,
        )
        col_datetime_start, col_datetime_end = (
            "measurement_datetime",
            "measurement_datetime",
        )
    elif t == observation:
        cols = select(
            t.c.observation_datetime.label("start_datetime"),
            t.c.value_as_number.label("value"),
        )
        col_datetime_start, col_datetime_end = (
            "observation_datetime",
            "observation_datetime",
        )
    elif t == drug_exposure:
        cols = select(
            t.c.drug_exposure_start_datetime.label("start_datetime"),
            t.c.drug_exposure_end_datetime.label("end_datetime"),
            t.c.quantity.label("value"),
            t.c.route_concept_id,
        )
        col_datetime_start, col_datetime_end = (
            "drug_exposure_start_datetime",
            "drug_exposure_end_datetime",
        )
    elif t == visit_occurrence:
        cols = select(
            t.c.visit_start_datetime.label("start_datetime"),
            t.c.visit_end_datetime.label("end_datetime"),
        )
        col_datetime_start, col_datetime_end = (
            "visit_start_datetime",
            "visit_end_datetime",
        )
    elif t == condition_occurrence:
        cols = select(
            t.c.condition_start_datetime.label("start_datetime"),
            t.c.condition_end_datetime.label("end_datetime"),
        )
        col_datetime_start, col_datetime_end = (
            "condition_start_datetime",
            "condition_end_datetime",
        )
    elif t == procedure_occurrence:
        cols = select(
            t.c.procedure_datetime.label("start_datetime"),
            t.c.procedure_end_datetime.label("end_datetime"),
        )
        col_datetime_start, col_datetime_end = (
            "procedure_datetime",
            "procedure_end_datetime",
        )
    else:
        raise HTTPException(status_code=400, detail=f"No columns mapped for table {t}")

    query = cols.where(
        and_(
            t.c["person_id"] == person_id,
            t.c[col_datetime_end] >= start_datetime,
            t.c[col_datetime_start] <= end_datetime,
            t.c[col_concept] == concept_id,
        )
    )

    # Execute query and return results
    result = db.execute(query)
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


#########################################
# Person Data (for OMOP viewer)


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
        query = text(
            f"""
            SELECT t.*, c.concept_name, c.concept_id
            FROM {data_schema}.{table_name} as t
            JOIN {data_schema}.concept c ON t.{table['concept_id']} = c.concept_id
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
    columns = []
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


@app.get("/full_day_coverage/{run_id}/", response_model=List[DayCoverage])
def get_full_day_coverage_endpoint(
    run_id: int,
    person_id: int | None = None,
    person_source_value: int | None = None,
    valid_date: datetime.date = Query(
        ..., description="The specified date to filter results."
    ),
    n_days: int = 10,
    db: Session = Depends(get_db),
) -> list[DayCoverage]:
    """
    Get full-day coverage intervals and combine with partial-day coverage entries
    (filtered for cohort_category = 'POPULATION'),
    including dates up to `n_days` before the specified valid_date.
    """
    if person_id is not None and person_source_value is not None:
        raise HTTPException(
            status_code=400,
            detail="Only one of person_id or person_source_value can be provided",
        )

    if n_days > 10 or n_days < 1:
        raise HTTPException(
            status_code=400, detail="n_days must be between 1 and 10 (inclusive)"
        )

    # Calculate date range
    date_lower_bound = valid_date - datetime.timedelta(days=n_days)

    # Base filters
    base_filters = and_(
        full_day_coverage.c.run_id == run_id,
        full_day_coverage.c.valid_date.between(date_lower_bound, valid_date),
        or_(
            and_(
                full_day_coverage.c.criterion_id.is_(None),
                full_day_coverage.c.pi_pair_id.is_(None),
                full_day_coverage.c.cohort_category == "POPULATION_INTERVENTION",
            ),
            full_day_coverage.c.cohort_category == "BASE",
        ),
    )

    partial_filters = and_(
        partial_day_coverage.c.run_id == run_id,
        partial_day_coverage.c.valid_date.between(date_lower_bound, valid_date),
        partial_day_coverage.c.cohort_category == "POPULATION",
    )

    # Add person filters
    if person_id:
        base_filters = and_(base_filters, full_day_coverage.c.person_id == person_id)
        partial_filters = and_(
            partial_filters, partial_day_coverage.c.person_id == person_id
        )
    elif person_source_value:
        base_filters = and_(
            base_filters,
            full_day_coverage.c.person_id == person.c.person_id,
            person.c.person_source_value == person_source_value,
        )
        partial_filters = and_(
            partial_filters,
            partial_day_coverage.c.person_id == person.c.person_id,
            person.c.person_source_value == person_source_value,
        )

    # Build queries for full_day and partial_day
    full_day_query = select(
        full_day_coverage.c.person_id,
        full_day_coverage.c.cohort_category,
        func.date(full_day_coverage.c.valid_date).label("valid_date"),
    ).where(base_filters)

    partial_day_query = select(
        partial_day_coverage.c.person_id,
        partial_day_coverage.c.cohort_category,
        func.date(partial_day_coverage.c.valid_date).label("valid_date"),
    ).where(partial_filters)

    # Combine queries using UNION
    combined_query = union(full_day_query, partial_day_query)

    # Execute the combined query
    result = db.execute(combined_query)
    rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No coverage data found for run_id {run_id} within the specified date range.",
        )
    # Convert rows into the response model
    return rows


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec (all interfaces is fine)
