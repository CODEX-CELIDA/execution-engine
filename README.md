CELIDA Execution Engine
============================

Execution engine for the CELIDA project.

Starting from machine-readable recommendations in CPG-on-EBM-on-FHIR format, this package generates OMOP
Cohort Definitions and executes them on a target OMOP CDM database.

Usage
-----

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
e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)
```

The results are written in the OMOP database (see [Configuration](#configuration)) in the schema `celida`.

### Configuration

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
