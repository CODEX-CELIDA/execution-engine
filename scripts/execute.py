"""
This script is designed to execute a set of medical recommendations from specified URLs
using the ExecutionEngine from the execution_engine package. It works with guidelines
related to COVID-19 and sepsis treatments from the 'netzwerk-universitaetsmedizin.de'
FHIR (Fast Healthcare Interoperability Resources) server.

The script defines a list of endpoint URLs for specific medical recommendations. It then
sets up a time range for the execution, using the pendulum library for date and time
handling. The execution is carried out for each recommendation URL within the specified
time range.

The ExecutionEngine is used to load and execute each recommendation. Logging is set up
to debug level for detailed information about the execution process.

Attributes:
    base_url (str): Base URL for the FHIR server from where recommendations are fetched.
    urls (list of str): Specific endpoints for medical recommendations to be executed.
    start_datetime (pendulum.DateTime): Start datetime for the execution range.
    end_datetime (pendulum.DateTime): End datetime for the execution range.
    e (ExecutionEngine): Instance of the ExecutionEngine used for executing recommendations.
    logger (logging.Logger): Logger for outputting the status and results of the executions.

Example:
    This script can be executed directly from the command line:
    $ python execute.py

Note:
    - This script assumes the presence of the 'execution_engine' package and its ExecutionEngine class.
    - The pendulum library is required for date and time handling.
    - This script is configured to fetch data from a specific base URL and may need modifications
      to work with other servers or data sources.
    - The time range for execution is hardcoded and may need adjustments as per requirements.
    - The execution_engine package expects a couple of environment variables to be set (see README.md).
"""

import logging
import os
import re
import sys

import pendulum
from sqlalchemy import text

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

from execution_engine.clients import omopdb
from execution_engine.execution_engine import ExecutionEngine
from execution_engine.settings import get_config, update_config

# enable multiprocessing with all available cores
update_config(multiprocessing_use=True, multiprocessing_pool_size=-1)

result_schema = get_config().omop.db_result_schema

# Validate the schema name to ensure it's safe to use in the query
if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", result_schema):
    raise ValueError(f"Invalid schema name: {result_schema}")

# Optional: Truncate all tables before execution
with omopdb.begin() as con:
    schema_exists = (
        con.execute(
            text(
                "SELECT count(*) FROM information_schema.schemata WHERE schema_name = :schema_name;"
            ),
            {"schema_name": result_schema},
        ).fetchone()[0]
        > 0
    )

    # If the schema exists, proceed to truncate tables
    if schema_exists:
        con.execute(
            text(
                "TRUNCATE TABLE "
                f"   {result_schema}.comment, "
                f"   {result_schema}.recommendation, "
                f"   {result_schema}.criterion, "
                f"   {result_schema}.execution_run, "
                f"   {result_schema}.result_interval, "
                f"   {result_schema}.recommendation, "
                f"   {result_schema}.population_intervention_pair "
                "RESTART IDENTITY",
            )
        )

base_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
recommendation_package_version = "latest"

urls = [
    "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation",
    "sepsis/recommendation/ventilation-plan-ards-tidal-volume",
    "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume",
    "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-peep",
    "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation",
    "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation",
    "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards",
]

start_datetime = pendulum.parse("2020-01-01 00:00:00+01:00")
end_datetime = pendulum.parse("2023-05-31 23:59:59+01:00")

e = ExecutionEngine()
logging.getLogger().setLevel(logging.DEBUG)


for recommendation_url in urls:
    print(recommendation_url)
    cdd = e.load_recommendation(
        base_url + recommendation_url,
        recommendation_package_version=recommendation_package_version,
    )

    e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)
