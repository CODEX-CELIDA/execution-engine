CELIDA Execution Engine
============================
### Integrate machine-readable clinical guideline recommendations with patient data

[![pytest](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml/badge.svg)](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/CODEX-CELIDA/execution-engine/branch/main/graph/badge.svg?token=XKACAB96VQ)](https://codecov.io/github/CODEX-CELIDA/execution-engine)



Starting from machine-readable recommendations in CPG-on-EBM-on-FHIR format, this package provides an execution engine
that can be used to execute the recommendations on patient data in OMOP CDM format.

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

3. Create an .env file (see [Configuration](#configuration)) or set the environment variables in your shell.

   You can copy the supplied `sample.env` file to `.env` and adjust the variables according to your local setup.
   ```bash
   cp .env.sample .env
   ```

4. Optionally: Compile the Cython code
    ```bash
    python setup.py build_ext --inplace
    ```

5. Run the following code

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

The results are written in the OMOP database (see [Configuration](#configuration)) in the schema `celida` (or the schema specified
in the `OMOP__RESULT_SCHEMA` environment variable).

Use the `interval_result`, `interval_coverage` and `full_day_coverage` views of the `celida` schema to analyse the
results of the cohort definition execution.

> [!WARNING]
> In contrast to the OMOP CDM 5.4 specification, the OMOP CDM database used for the execution engine
> **must** implement all `*_datetime` fields as `TIMESTAMP WITH TIME ZONE NOT NULL`.


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
OMOP__DATA_SCHEMA=cds_cdm

## Execution Engine Result Schema
OMOP__RESULT_SCHEMA=celida

# Execution Engine Configuration
## Timezone used for date/time calculations
CELIDA_EE_TIMEZONE=Europe/Berlin

## Episode of Care Mapping Table
## Set 1 to Use VISIT_DETAIL for episode of care mappings instead of VISIT_OCCURRENCE
CELIDA_EE_EPISODE_OF_CARE_VISIT_DETAIL=0

# Parallel processing options
# Set 1 if multiprocessing (parallelization) should be used, 0 otherwise
CELIDA_EE_MULTIPROCESSING_USE=0

# Set number of workers in multiprocessing pool. Use -1 to use number of available cpu cores.
CELIDA_EE_MULTIPROCESSING_POOL_SIZE=-1

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

### 4. Start Recommendation Server
See [CODEX-CELIDA Recommendation Server on GitHub.](https://github.com/CODEX-CELIDA/recommendation-server)

### 5. Run Pytest

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
Optionally, add the `--run-slow-tests` flag to run extensive test cases for the recommendations.

This will execute the tests with the specified PostgreSQL configuration and additional options for the test run.
