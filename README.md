CELIDA Execution Engine
============================
### Integrate machine-readable clinical guideline recommendations with patient data

[![pytest](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml/badge.svg)](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/CODEX-CELIDA/execution-engine/branch/main/graph/badge.svg?token=XKACAB96VQ)](https://codecov.io/github/CODEX-CELIDA/execution-engine)



Starting from machine-readable recommendations in CPG-on-EBM-on-FHIR format, this package generates OMOP
Cohort Definitions and executes them on a target OMOP CDM database.

Usage
-----

### Standalone

The following example shows how to generate a cohort definition from a recommendation in [CPG-on-EBM-on-FHIR format](https://ceosys.github.io/cpg-on-ebm-on-fhir/)
and execute it on a target OMOP CDM database.

```python
from datetime import datetime, timedelta

from execution_engine import ExecutionEngine

# Specify canonical URL of recommendation to process
base_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
recommendation_url = "sepsis/recommendation/ventilation-plan-ards-tidal-volume"


# Set the time window of patient data that should be checked against the recommendation
# This example uses the past week
end_datetime = datetime.today()
start_datetime = end_datetime - timedelta(days=7)

# Initialize execution engine
e = ExecutionEngine()

# Load recommendation
cdd = e.load_recommendation(base_url + recommendation_url)

# Execute recommendation
run_id = e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)

# Get results
e.fetch_patients(run_id)
e.fetch_criteria(run_id)
e.fetch_patient_data(person_id, criterion_name, cdd, start_datetime, end_datetime)
```

The results are written in the OMOP database (see [Configuration](#configuration)) in the schema `celida`.

### FastAPI Web Service

```bash

# Start web service
$ uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
```

Open Swagger UI at http://localhost:8000/docs for documentation.

Try out the endpoints:
#### Get all recommendations

```python
import requests
import pendulum # for datetime handlings

r = requests.get("http://localhost:8001/recommendation/list")

r.json() # returns list of recommendations
```

#### Get the criteria associated with a single recommendation

```python
url = 'https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation'

r = requests.get("http://localhost:8001/recommendation/criteria", params={"recommendation_url": url})

r.json() # list of criteria

# Execute a recommendation
end_datetime = pendulum.now()
start_datetime = end_datetime.add(-7)

params={
    "recommendation_url":url,
    "start_datetime": start_datetime.for_json(),
    "end_datetime": end_datetime.for_json()
}

r = requests.get("http://192.168.200.128:8001/patient/list", params=params)

r.json() # list of patients that match the criteria
```

#### Get individual patient data for the previously executed recommendation
```python
run_id = r.json()["run_id"]

params={
    "run_id": int(run_id),
    "person_id": "123", # one of the person_ids from the previous request
    "criterion_name": "test", # one of the above criterion names
}

r = requests.get("http://192.168.200.128:8001/patient/data", params=params)
r.json()
```


## Configuration

The following environment variables need to be defined (e.g., in a .env file):

``` bash
# FHIR Configuration
## The FHIR server serving CPG-on-EBM-on-FHIR resources
FHIR_BASE_URL=http://localhost:8000/fhir

## A FHIR terminology server
FHIR_TERMINOLOGY_SERVER_URL=http://tx.fhir.org/r4

# OMOP Configuration
## OMOP Database Username
OMOP__USER=postgres

## OMOP Database Password
OMOP__PASSWORD=<your_password>

## OMOP Database Host
OMOP__HOST=localhost

## OMOP Database Port
OMOP__PORT=5432

## OMOP Database
OMOP__DATABASE=ohdsi

## OMOP Database Schema
OMOP__SCHEMA=cds_cdm
```

You can copy the supplied `sample.env` file to `.env` and adjust the variables according to your local setup.
