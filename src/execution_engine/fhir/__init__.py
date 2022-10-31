import os

from fhir.resources.codeableconcept import CodeableConcept

from ..execution_engine import LOINC, SNOMEDCT
from .client import FHIRClient
from .terminology import FHIRTerminologyClient

tx_client = FHIRTerminologyClient(os.environ["FHIR_TERMINOLOGY_SERVER_URL"])
fhir_client = FHIRClient(os.environ["FHIR_BASE_URL"])


def isSNOMEDCT(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a SNOMED CT concept."""
    return cc.system == SNOMEDCT


def isLOINC(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a LOINC concept."""
    return cc.system == LOINC
