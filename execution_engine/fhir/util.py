from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding


def get_coding(cc: CodeableConcept) -> Coding:
    """
    Get the first (and only one) coding from a CodeableConcept.
    """
    assert len(cc.coding) == 1, "CodeableConcept must have exactly one coding"
    return cc.coding[0]
