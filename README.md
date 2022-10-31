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
from execution_engine.execution_engine import ExecutionEngine

# set environment variables (or use a .env file)
os.environ["FHIR_BASE_URL"] = "http://localhost:8000/fhir"
os.environ["OMOP_WEBAPI_URL"] = "http://192.168.200.128:9876/WebAPI"
os.environ["FHIR_TERMINOLOGY_SERVER_URL"] = "http://tx.fhir.org/R4"

# canonical URL of a recommendation in CPG-on-EBM-on-FHIR format
recommendation_url = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/recommendations/intervention-plan/antithrombotic-prophylaxis-LMWH"

e = ExecutionEngine()

# load recommendation from FHIR server and convert it into OMOP cohort definition
cohort_def = e.process_recommendation(recommendation_url)

# create cohort in OMOP CDM database
e.create_cohort(
    name='codex-celida-recommendation-cohort',
    description=f'Cohort for the CODEX-CELIDA recommendation found at {recommendation_url}',
    cohort_def
 )
