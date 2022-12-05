CELIDA Execution Engine
============================

Execution engine for the CELIDA project.

Starting from machine-readable recommendations in CPG-on-EBM-on-FHIR format, this package generates OMOP
Cohort Definitions and executes them on a target OMOP CDM database.

Example
-------

The following example shows how to generate a cohort definition from a recommendation in [CPG-on-EBM-on-FHIR format](https://ceosys.github.io/cpg-on-ebm-on-fhir/)
and execute it on a target OMOP CDM database.

```python
import os
from execution_engine import ExecutionEngine

# set environment variables (or use a .env file)
os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/R4"
os.environ["OMOP_DB_HOST"] = "localhost"
os.environ["OMOP_DB_PORT"] = "5432"
os.environ["OMOP_DB_NAME"] = "ohdsi"
os.environ["OMOP_DB_USER"] = "postgres"
os.environ["OMOP_DB_PASSWORD"] = "postgres"
os.environ["OMOP_DB_SCHEMA"] = "cds_cdm"

# canonical URL of a recommendation in CPG-on-EBM-on-FHIR format
recommendation_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/intervention-plan/antithrombotic-prophylaxis-LMWH"

e = ExecutionEngine()

# load recommendation from FHIR server and convert it into OMOP cohort definition
cohort_def = e.process_recommendation(recommendation_url)

# get series of SQL statements for selecting patients from OMOP CDM according to the cohort definition
for statement in cd.process():
    print(omopdb.compile_query(statement))
