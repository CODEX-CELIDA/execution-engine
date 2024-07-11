# CELIDA Execution Engine

 Integrate machine-readable clinical guideline recommendations with patient data.

[![pytest](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml/badge.svg)](https://github.com/CODEX-CELIDA/execution-engine/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/CODEX-CELIDA/execution-engine/branch/main/graph/badge.svg?token=XKACAB96VQ)](https://codecov.io/github/CODEX-CELIDA/execution-engine)



Starting from machine-readable recommendations in CPG-on-EBM-on-FHIR format, this package provides an execution engine
that can be used to execute the recommendations on patient data in OMOP CDM format.

## Usage

### Requirements

#### Python
- Python 3.11

#### FHIR Server
- A running FHIR server serving CPG-on-EBM-on-FHIR resources (e.g. the [CELIDA recommendation server](https://github.com/CODEX-CELIDA/recommendation-server)).

#### OMOP CDM Database
 - A **PostgreSQL** server running the [OMOP CDM 5.4](https://ohdsi.github.io/CommonDataModel/cdm54.html) database
 - The database must contain all relevant concepts from the [OMOP Vocabulary](https://athena.ohdsi.org/vocabulary/list)
 - The database must be accessible from the machine running the execution engine
 - The database must be populated with patient data in OMOP CDM format

> [!WARNING]
> In contrast to the OMOP CDM 5.4 specification, the OMOP CDM database used for the execution engine
> **must** implement all `*_datetime` fields as `TIMESTAMP WITH TIME ZONE NOT NULL`.


### Setup
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
   cp sample.env .env
   ```

4. Optionally: Compile the Cython code
    ```bash
    python setup.py build_ext --inplace
    ```

### Execute Recommendations
 Run the following code (also present in [scripts/execute.py](https://github.com/CODEX-CELIDA/execution-engine/blob/main/scripts/execute.py)):

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

## Results of the Execution

Below is an overview of how the data is structured and how to interpret the results.

### Data Storage and Schema

- **Result Storage**: The results are stored in the `celida.result_interval` table of the OMOP database.
  Refer to the [Configuration](#configuration) section for more details.
- **Schema Customization**: By default, the results are written in the `celida` schema. This can be altered
  using the `CELIDA_EE_OMOP__RESULT_SCHEMA` environment variable.

### Recommendation Process

The recommendation process is divided into several steps:

1. **Criteria Definition**:
   - The execution engine evaluates distinct criteria such as "COVID-19", "Ventilated", "FiO2 between 60-70%", or
     "Tidal volume <= 6 ml/kg ideal body weight".

2. **Subpopulation Combination ("Population/Intervention Pairs")**:
   - These criteria are combined into subpopulations, e.g., "COVID-19 and FiO2 between 60-69.9%",
     "COVID-19 and FiO2 between 70-79.9%".

3. **Overall Recommendation Composition**:
   - The subpopulations are aggregated into the overall recommendation.

4. **Categorization**:
   - Each criterion and subpopulation is categorized as either "POPULATION", "INTERVENTION",
     or "POPULATION_INTERVENTION" in the cohort_category column.
   - "POPULATION_INTERVENTION" is used for the combination of a population with its corresponding intervention.

5. **Base Criterion**:
   - A BASE criterion is established to filter all patients in the database, typically selecting active patients
     during the observation period. This is reflected in the BASE cohort_category.

6. **Result Combination**:
   - Post-evaluation, results are combined using AND, OR, and NOT operators based on the recommendation definition.

### Result Intervals

- **Structure**: A result interval comprises an `interval_start`, `interval_end`, and `interval_type` , describing the
  coverage of a criterion or combination.
- **Interval Types**:
  - POSITIVE: Criterion/combination is fulfilled.
  - NEGATIVE: Criterion/combination is not fulfilled.
  - NO_DATA: No data available to evaluate the criterion/combination.
  - NOT_APPLICABLE: Not applicable, used for POPULATION_INTERVENTION where POPULATION is NEGATIVE.

### Execution Run

Each execution run is registered in the `celida.execution_run` table and identified by a `run_id`.
This `run_id` is used to identify the results of the execution in the `celida.result_interval` table.
Note that if you run the same recommendation multiple times, each run will be stored in the database with
its own `run_id`. That means that results should be retrieved using the `run_id` of the execution run.

### Convenience Views for Analysis

To facilitate result analysis, the following views are available:

- `celida.interval_result`: Same structure as `celida.result_interval` but with additional columns for the
  recommendation name, the criterion description, and the subpopulation ("pi_pair") name.
- `celida.interval_coverage`: One row per day within the observation period (defined in the execution run) and criterion
  or combination of criteria instead of one row per interval and criterion. For each day, the columns `has_positive`,
  `has_negative`, and `has_no_data` indicate whether there is at least one interval of the corresponding type
  that overlaps fully or partially with the day. The column `covered_time` contains the duration that is covered by any
  interval.
- `celida.full_day_coverage`: Filter on `celida.interval_coverage` to show only days where the
  criterion or combination of criteria is fulllfilled during the whole day. Fulfilled here means that
    (i) the criterion/combination has at least one positive interval that overlaps (partially or fully) with the day and
    (ii) there is no negative interval that overlaps with the day.
  This view is useful for analysing the coverage of the recommendation over time and can be considered the final result.
- `celida.partial_day_coverage`: Not currently used.

To analyze the results of the cohort definition execution, primarily use the `interval_result`, `interval_coverage`,
and `full_day_coverage` views in the `celida` schema.


#### Summary of Results
The following query yields a summary of the results for each recommendation and execution run, aggregated by
cohort category (POPULATION, INTERVENTION, POPULATION_INTERVENTION) and day:
```sql
SELECT
    rec.recommendation_name,
    run.run_id,
    res.cohort_category,
    res.valid_date,
    count(*)
FROM celida.full_day_coverage res
INNER JOIN celida.execution_run run on (run.run_id = res.run_id)
INNER JOIN celida.recommendation cd ON (rec.recommendation_id = run.recommendation_id)
WHERE
    res.pi_pair_id IS NULL  -- pi_pair_id identifies subpopulations in the recommendation
                            -- and is NULL for the overall recommendation
  AND (
      res.criterion_id IS NULL -- criterion_id identifies single criteria in the recommendation
                               -- and is NULL for the overall recommendation
      or res.cohort_category = 'BASE' -- BASE category contains all patients from the "base criterion", i.e.
                                      -- all patients that are active during the requested observation period of
                                      -- the execution run
  )
GROUP BY rec.recommendation_name, run.run_id, res.cohort_category, res.valid_date
```

> [!NOTE]
> This query is not yet optimized and may run for several minutes for large datasets (e.g., 10k patients over 3 years).


#### Individual Results
The following query yields the individual results for a specific patient and recommendation:

```sql
SELECT
    run.run_id,
    rec.recommendation_name,
    rec.recommendation_url,
    res.pi_pair_name, -- name of the subpopulation in the recommendation
    res.criterion_description, -- description of the criterion in the recommendation
    res.cohort_category, -- category of the criterion in the recommendation (BAS, POPULATION, INTERVENTION, POPULATION_INTERVENTION)
    res.interval_start, -- start of a time interval describing the coverage of the criterion or combination of criteria
    res.interval_end, -- end of a time interval
    res.interval_type -- type of the interval:
                       --   POSITIVE (criterion or combination is fulfilled)
                       --   NEGATIVE (criterion or combination is not fulfilled)
                       --   NO_DATA (no data available for the criterion or combination)
                       --   NOT_APPLICABLE (criterion or combination is not applicable
                       --       - only used for POPULATION_INTERVENTION where the POPULATION is NEGATIVE)
FROM celida.interval_result res
INNER JOIN celida.execution_run run on (run.run_id = res.run_id)
INNER JOIN celida.recommendation rec ON (rec.recommendation_id = run.recommendation_id)
WHERE
    res.person_id = 1234567890
    AND rec.recommendation_url = 'covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation'
    -- or filter by run_id
```

## Configuration

The following environment variables need to be defined (e.g., in a .env file):

```env
# FHIR Configuration
## The FHIR server serving CPG-on-EBM-on-FHIR resources
CELIDA_EE_FHIR_BASE_URL=http://localhost:8000/fhir

## A FHIR terminology server
CELIDA_EE_FHIR_TERMINOLOGY_SERVER_URL=http://tx.fhir.org/r4

# OMOP Configuration
## OMOP Database Username
CELIDA_EE_OMOP__USER=postgres

## OMOP Database Password
CELIDA_EE_OMOP__PASSWORD=<your_password>

## OMOP Database Host
CELIDA_EE_OMOP__HOST=localhost

## OMOP Database Port
CELIDA_EE_OMOP__PORT=5432

## OMOP Database
CELIDA_EE_OMOP__DATABASE=ohdsi

## OMOP Database Schema
CELIDA_EE_OMOP__DATA_SCHEMA=cds_cdm

## Execution Engine Result Schema
CELIDA_EE_OMOP__RESULT_SCHEMA=celida
CELIDA_EE_OMOP__RESULT_SCHEMA=celida

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

# Time Intervals (used for shift definitions)
## Morning Shift
CELIDA_EE_TIME_INTERVALS__MORNING_SHIFT__START=06:00:00
CELIDA_EE_TIME_INTERVALS__MORNING_SHIFT__END=13:59:59

## Afternoon Shift
CELIDA_EE_TIME_INTERVALS__AFTERNOON_SHIFT__START=14:00:00
CELIDA_EE_TIME_INTERVALS__AFTERNOON_SHIFT__END=21:59:59

## Night Shift
CELIDA_EE_TIME_INTERVALS__NIGHT_SHIFT__START=22:00:00
CELIDA_EE_TIME_INTERVALS__NIGHT_SHIFT__END=05:59:59

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
