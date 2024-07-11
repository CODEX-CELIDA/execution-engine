from abc import ABC, abstractmethod
from typing import Tuple, Type

from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.element import Element
from fhir.resources.extension import Extension
from fhir.resources.fhirtypes import Boolean
from fhir.resources.quantity import Quantity
from fhir.resources.range import Range

from execution_engine.fhir.util import get_coding
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.vocabulary import standard_vocabulary
from execution_engine.util.value import Value, ValueConcept, ValueNumber


@staticmethod
def select_value(
    root: Element, value_prefix: str
) -> CodeableConcept | Quantity | Range | Boolean:
    """
    Selects the value of a characteristic by datatype.
    """
    for datatype in ["CodeableConcept", "Quantity", "Range", "Boolean"]:
        element_name = f"{value_prefix}{datatype}"
        value = getattr(root, element_name, None)
        if value is not None:
            return value
    raise ValueError("No value found")


def parse_code_value(
    code: CodeableConcept, value_parent: Element, value_prefix: str
) -> Tuple[Concept, Value]:
    """
    Parses a code and value from a FHIR CodeableConcept into OMOP concepts and values.
    """
    return parse_code(code), parse_value(value_parent, value_prefix)


def parse_code(code: CodeableConcept) -> Concept:
    """
    Parses a FHIR code into a standard OMOP concept.
    """
    cc = get_coding(code)
    return standard_vocabulary.get_standard_concept(cc.system, cc.code)


def code_display(code: CodeableConcept) -> str:
    """
    Get the display of a CodeableConcept (or the code alternatively).
    """
    cc = get_coding(code)

    if cc.display is not None:
        return cc.display

    return cc.code


def parse_value(value_parent: Element, value_prefix: str) -> Value:
    """
    Parses a value from a FHIR element.
    """
    value = select_value(value_parent, value_prefix)
    value_obj: Value

    if isinstance(value, CodeableConcept):
        cc = get_coding(value)
        value_omop_concept = standard_vocabulary.get_standard_concept(
            system_uri=cc.system, concept=cc.code
        )
        value_obj = ValueConcept(value=value_omop_concept)
    elif isinstance(value, Quantity):
        value_obj = ValueNumber(
            value=value.value,
            unit=standard_vocabulary.get_standard_unit_concept(value.code),
        )
    elif isinstance(value, Range):
        if value.low is not None and value.high is not None:
            assert (
                value.low.code == value.high.code
            ), "Range low and high unit must be the same"

        unit_code = value.low.code if value.low is not None else value.high.code

        def value_or_none(x: Quantity | None) -> float | None:
            if x is None:
                return None
            return x.value

        value_obj = ValueNumber(
            unit=standard_vocabulary.get_standard_unit_concept(unit_code),
            value_min=value_or_none(value.low),
            value_max=value_or_none(value.high),
        )
    else:
        raise NotImplementedError(f"Value type {type(value)} not implemented")

    return value_obj


class CriterionConverter(ABC):
    """
    An abstract criterion converter (interface).

    An instance of this class performs the conversion of some FHIR element to an OMOP criterion.
    """

    def __init__(self, exclude: bool):
        # todo: is exclude still required?
        self._exclude = exclude

    @classmethod
    @abstractmethod
    def from_fhir(cls, fhir_definition: Element) -> "CriterionConverter":
        """Creates a new Converter from a FHIR element/resource."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def valid(cls, fhir_definition: Element) -> bool:
        """
        Checks if the given FHIR definition is valid in the context of CPG-on-EBM-on-FHIR.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_criterion(self) -> Criterion | LogicalCriterionCombination:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()


class CriterionConverterFactory:
    """Factory for creating a new instance of a criterion converter."""

    def __init__(self) -> None:
        self._converters: list[Type[CriterionConverter]] = []

    def register(self, converter: Type[CriterionConverter]) -> None:
        """Register a new characteristic type."""
        self._converters.append(converter)

    def get(self, fhir: Element) -> CriterionConverter:
        """
        Get a converter for the given FHIR element.
        """
        for converter in self._converters:
            if converter.valid(fhir):
                return converter.from_fhir(fhir)

        message = f"Cannot find a converter for the given FHIR element: {fhir.__class__.__name__}"
        if fhir.id is not None:
            message += f' (id="{fhir.id}")'
        raise ValueError(message)


def get_extension_by_url(element: Element, url: str) -> Extension:
    """
    Retrieves an Extension object from a list of extensions based on the provided URL.

    Args:
        element: The FHIR element containing the extensions.
        url (str): The URL to match in the Extension objects.

    Returns:
        Optional[Extension]: The Extension object matching the URL if found, else None.

    Raises:
        ValueError: If no extensions match the URL or more than one extension matches the URL.
    """
    matching_extensions = [ext for ext in element.extension if ext.url == url]

    if len(matching_extensions) == 0:
        raise ValueError(f"No extension found with URL: {url}")
    elif len(matching_extensions) > 1:
        raise ValueError(
            f"Multiple extensions found with URL: {url}, expected only one."
        )

    return matching_extensions[0]
