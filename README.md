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


The following example shows how to generate a cohort definition from multiple recommendations in [CPG-on-EBM-on-FHIR format](https://ceosys.github.io/cpg-on-ebm-on-fhir/)
and execute it on a target OMOP CDM database.

1. Clone this git repository
    ```bash
    git clone https://github.com/CODEX-CELIDA/execution-engine
    ```

2. Setup conda (or venv) environment and install requirements (Python 3.11 required):
    ```bash
    conda create -n execution-engine python=3.11
    conda activate execution-engine
    pip install -r requirements.txt
    ```

3. Create an .env file (see [Configuration](#configuration))

4. Run the following code

   ```python
   import pendulum
   import logging

   import os
   os.chdir("/path/to/execution-engine-git-repository")

   from execution_engine.execution_engine import ExecutionEngine

   base_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"

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
       cdd = e.load_recommendation(base_url + recommendation_url)

       e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)
   ```

The results are written in the OMOP database (see [Configuration](#configuration)) in the schema `celida`.

### FastAPI Web Service

> [!WARNING]
> The FastAPI Web Service documentation is outdated

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

# Execution Engine Configuration
CELIDA_EE_TIMEZONE=Europe/Berlin
```

You can copy the supplied `sample.env` file to `.env` and adjust the variables according to your local setup.

## Testing

Follow these steps to set up and run tests for the project:

### 1. Clone Submodules

First, ensure that you have cloned the submodules to get the necessary test data:

```bash
git submodule update --init --recursive
```

### 2. Start PostgreSQL Container

Ensure a PostgreSQL container is running. You can do this manually 

```bash
docker run \
  --name postgres-pytest-celida \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5434:5432 \
  -d postgres
```

or by using the provided script:

```bash
./start-postgres-pytest.sh
```

### 3. Install Development Requirements

Install the necessary packages for testing from requirements-dev.txt:

```bash
pip install -r requirements-dev.txt
```

### 4. Run Pytest

Finally, run pytest with the necessary parameters:

```bash
pytest \
  --postgresql-host=localhost \
  --postgresql-port=5434 \
  --postgresql-user=postgres \
  --postgresql-password=mysecretpassword \
  --color=yes \
  --run-recommendation-tests
```

This will execute the tests with the specified PostgreSQL configuration and additional options for the test run.

