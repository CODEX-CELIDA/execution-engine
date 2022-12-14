import pandas as pd
from pydantic import BaseModel


class Concept(BaseModel, frozen=True):  # type: ignore
    """Represents an OMOP Standard Vocabulary concept."""

    id: int
    name: str
    concept_code: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str | None = None
    invalid_reason: str | None = None

    @staticmethod
    def from_dict(data: dict) -> "Concept":
        """Creates a concept from a dict representation."""
        return Concept(
            id=data["CONCEPT_ID"],
            name=data["CONCEPT_NAME"],
            domain_id=data["DOMAIN_ID"],
            vocabulary_id=data["VOCABULARY_ID"],
            concept_code=data["CONCEPT_CODE"],
            concept_class_id=data["CONCEPT_CLASS_ID"],
            standard_concept=data.get("STANDARD_CONCEPT"),
            invalid_reason=data.get("INVALID_REASON"),
        )

    @staticmethod
    def from_series(series: pd.Series) -> "Concept":
        """Creates a concept from a pandas Series."""
        return Concept.from_dict({k.upper(): v for k, v in series.to_dict().items()})

    def dict(self) -> dict:  # type: ignore
        """Returns a dict representation of the concept."""
        return {
            "CONCEPT_ID": self.id,
            "CONCEPT_NAME": self.name,
            "DOMAIN_ID": self.domain_id,
            "VOCABULARY_ID": self.vocabulary_id,
            "CONCEPT_CODE": self.concept_code,
            "CONCEPT_CLASS_ID": self.concept_class_id,
            "STANDARD_CONCEPT": self.standard_concept,
            "INVALID_REASON": self.invalid_reason,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the concept.
        """
        base = f'OMOP Concept: "{self.name}" ({self.id}) [{self.vocabulary_id}#{self.concept_code}]'

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
        return self.id < 0


class CustomConcept(Concept):
    """Represents a custom concept."""

    def __init__(
        self, name: str, concept_code: str, domain_id: str, vocabulary_id: str
    ) -> None:
        """Creates a custom concept."""
        super().__init__(
            id=-1,
            name=name,
            concept_code=concept_code,
            domain_id=domain_id,
            vocabulary_id=vocabulary_id,
            concept_class_id="Custom",
            standard_concept=None,
            invalid_reason=None,
        )

    @property
    def id(self) -> int:  # type: ignore # todo: fix this (Signature of "id" incompatible with supertype "Concept")
        """
        Returns the concept id.
        """
        raise ValueError("Custom concepts do not have an id")

    def __str__(self) -> str:
        """
        Returns a string representation of the concept.
        """
        return (
            f'Custom Concept: "{self.name}" [{self.vocabulary_id}#{self.concept_code}]'
        )
