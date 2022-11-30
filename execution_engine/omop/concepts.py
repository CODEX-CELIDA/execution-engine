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
    def from_json(json: dict) -> "Concept":
        """Creates a concept from a JSON representation."""
        return Concept(
            id=json["CONCEPT_ID"],
            name=json["CONCEPT_NAME"],
            domain_id=json["DOMAIN_ID"],
            vocabulary_id=json["VOCABULARY_ID"],
            concept_code=json["CONCEPT_CODE"],
            concept_class_id=json["CONCEPT_CLASS_ID"],
            standard_concept=json.get("STANDARD_CONCEPT"),
            invalid_reason=json.get("INVALID_REASON"),
        )

    @staticmethod
    def from_series(series: pd.Series) -> "Concept":
        """Creates a concept from a pandas Series."""
        return Concept.from_json({k.upper(): v for k, v in series.to_dict().items()})

    def json(self) -> dict:
        """Returns a JSON representation of the concept."""
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


class ConceptSet:
    """A set of concepts.

    This class is used to represent a set of concepts. It can be used to
    represent a set of concepts in a cohort definition or a set of concepts
    in a concept set. The class is used to represent the concept set in
    the OMOP CDM.
    """

    def __init__(self, name: str, concept: Concept):
        """
        Initialize the concept set.

        Args:
          name: The name of the concept set.
          concept: The concept in the concept set.

        """
        self.id: int | None = None
        self.name = name
        self.concept = concept

    def json(self) -> dict:
        """Return the JSON representation of the concept set."""

        assert self.id is not None, "Concept set ID is None"

        return {
            "id": self.id,
            "name": self.name,
            "expression": {
                "items": [{"concept": self.concept.json(), "includeDescendents": True}]
            },
        }

    def __eq__(self, other: object) -> bool:
        """Return True if the concept sets are equal."""
        if not isinstance(other, ConceptSet):
            return False

        return self.name == other.name and self.concept == other.concept


class ConceptSetManager:
    """
    Manager for concept sets.

    Used to keep a set of unique concept sets and to automatically assign numeric, unique IDs to each concept set.
    """

    def __init__(self) -> None:
        self._concept_sets: list[ConceptSet] = []

    def add(self, concept_set: ConceptSet) -> ConceptSet:
        """
        Add a concept set to the manager.

        Args:
          concept_set: The concept set.

        Returns:
          The concept set.
        """

        if concept_set in self._concept_sets:
            return self._concept_sets[self._concept_sets.index(concept_set)]

        concept_set.id = len(self._concept_sets)

        self._concept_sets.append(concept_set)

        return concept_set

    def reset(self) -> None:
        """
        Reset the concept set manager.
        """
        self._concept_sets = []

    def json(self) -> list[dict]:
        """
        Return the JSON representation of the concept sets.
        """
        return [concept_set.json() for concept_set in self._concept_sets]
