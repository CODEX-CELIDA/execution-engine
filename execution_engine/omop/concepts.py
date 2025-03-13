import pandas as pd
from pydantic import BaseModel

from execution_engine.util import serializable


@serializable.register_class
class Concept(BaseModel, frozen=True):  # type: ignore
    """Represents an OMOP Standard Vocabulary concept."""

    concept_id: int
    concept_name: str
    concept_code: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str | None = None
    invalid_reason: str | None = None

    @staticmethod
    def from_series(series: pd.Series) -> "Concept":
        """Creates a concept from a pandas Series."""
        return Concept(**series.to_dict())

    def __str__(self) -> str:
        """
        Returns a string representation of the concept.
        """
        if self.vocabulary_id == "UCUM":
            return str(self.concept_code)

        return str(self.concept_name)

    def is_custom(self) -> bool:
        """
        Returns True if the concept is a custom concept (not a standard concept, i.e. not in the OMOP Standard Vocabulary).
        """
        return self.concept_id < 0


@serializable.register_class
class CustomConcept(Concept, frozen=True):
    """Represents a custom concept."""

    concept_id: int = -1
    concept_class_id: str = "Custom"
    standard_concept: str | None = None
    invalid_reason: str | None = None

    @property
    def id(self) -> int:
        """
        Returns the concept id.
        """
        raise ValueError("Custom concepts do not have an id")

    def __str__(self) -> str:
        """
        Returns a string representation of the concept.
        """
        return f'Custom Concept: "{self.concept_name}" [{self.vocabulary_id}#{self.concept_code}]'
