from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.element import Element


def get_coding(cc: CodeableConcept, system_uri: str | None = None) -> Coding:
    """
    Get the first (and only one) coding from a CodeableConcept.
    """
    if system_uri is None:
        coding = cc.coding
    else:
        coding = [c for c in cc.coding if c.system == system_uri]

    if cc.coding is None or len(coding) != 1:
        raise ValueError(
            f"CodeableConcept must have exactly one coding, got {len(coding) if cc.coding else 'None'}"
        )

    return coding[0]


def get_extension(base: Element, extension_url: str) -> Element | None:
    """
    Get the extension with the given URL from the given element.
    """
    if base.extension is None:
        return None

    for ext in base.extension:
        if ext.url == extension_url:
            return ext

    return None
