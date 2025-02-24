import json
from typing import Type

from fhir.resources.element import Element

from execution_engine.converter.time_from_event.abstract import TemporalIndicator


class TemporalIndicatorConverterFactory:
    """Factory for creating a new instance of a criterion converter."""

    def __init__(self) -> None:
        self._converters: list[Type[TemporalIndicator]] = []

    def register(self, converter: Type[TemporalIndicator]) -> None:
        """Register a new characteristic type."""
        self._converters.append(converter)

    def get(self, fhir: Element) -> TemporalIndicator:
        """
        Get a converter for the given FHIR element.
        """
        for converter in self._converters:
            if converter.valid(fhir):
                return converter.from_fhir(fhir)

        message = f"Cannot find a converter for the given FHIR element: {fhir.__class__.__name__}"
        if fhir.id is not None:
            message += f' (id="{fhir.id}")'

        message += f"\nFHIR element details: {json.dumps(fhir.model_dump(), indent=2)}"

        raise ValueError(message)
