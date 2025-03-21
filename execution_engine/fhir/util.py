from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.element import Element
from fhir.resources.extension import Extension


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


def get_extensions(base: Element, extension_url: str) -> list[Extension]:
    """
    Retrieves all extensions with the given URL from the specified element.

    This function returns a list of extensions that match the given URL,
    allowing for multiple occurrences.

    Args:
        base (Element): The element containing extensions.
        extension_url (str): The URL of the extensions to retrieve.

    Returns:
        list[Extension]: A list of matching extensions (empty if none are found).
    """
    if base.extension is None:
        return []

    return [ext for ext in base.extension if ext.url == extension_url]


def pop_extensions(base: Element, extension_url: str) -> list[Extension]:
    """
    Retrieves and removes all extensions with the given URL from the specified element,
    returning them as a list in the order they appeared.

    Args:
        base (Element): The element containing extensions.
        extension_url (str): The URL of the extensions to retrieve and remove.

    Returns:
        list[Extension]: A list of matching extensions (empty if none are found).
    """
    if not base.extension:
        return []

    matches = [ext for ext in base.extension if ext.url == extension_url]
    keepers = [ext for ext in base.extension if ext.url != extension_url]

    base.extension = keepers

    return matches


def get_extension(base: Element, extension_url: str) -> Extension | None:
    """
    Retrieves a single extension with the given URL from the specified element.

    If multiple extensions with the same URL exist, an error is raised to ensure
    that only unique extensions are retrieved.

    Args:
        base (Element): The element containing extensions.
        extension_url (str): The URL of the extension to retrieve.

    Returns:
        Extension | None: The matching extension if found, otherwise None.

    Raises:
        ValueError: If multiple extensions with the same URL exist.
    """
    matching_extensions = get_extensions(base, extension_url)

    if len(matching_extensions) > 1:
        raise ValueError(
            f"Multiple extensions found with URL '{extension_url}', but only one was expected."
        )

    return matching_extensions[0] if matching_extensions else None
