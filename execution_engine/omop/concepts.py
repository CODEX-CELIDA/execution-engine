import pandas as pd
from pydantic import BaseModel


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
        base = f'OMOP Concept: "{self.concept_name}" ({self.concept_id}) [{self.vocabulary_id}#{self.concept_code}]'

        if self.standard_concept is not None and self.standard_concept == "S":
            return f"{base} (STANDARD)"

        return base

    def __repr__(self) -> str:
        """
        Returns a string representation of the concept.
        """
        return str(self)

    def is_custom(self) -> bool:
        """
        Returns True if the concept is a custom concept (not a standard concept, i.e. not in the OMOP Standard Vocabulary).
        """
        return self.concept_id < 0


class CustomConcept(Concept):
    """Represents a custom concept."""

    def __init__(
        self, name: str, concept_code: str, domain_id: str, vocabulary_id: str
    ) -> None:
        """Creates a custom concept."""
        super().__init__(
            concept_id=-1,
            concept_name=name,
            concept_code=concept_code,
            domain_id=domain_id,
            vocabulary_id=vocabulary_id,
            concept_class_id="Custom",
            standard_concept=None,
            invalid_reason=None,
        )

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
