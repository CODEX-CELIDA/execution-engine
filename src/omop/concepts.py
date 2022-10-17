from typing import Dict, Optional, List


class ConceptSet:
    """A set of concepts.

    This class is used to represent a set of concepts. It can be used to
    represent a set of concepts in a cohort definition or a set of concepts
    in a concept set. The class is used to represent the concept set in
    the OMOP CDM.
    """

    def __init__(self, id: int, name: str, concept: Dict):
        """
        Initialize the concept set.

        Args:
          name: The name of the concept set.
          concept: The concept in the concept set.

        """
        self.id = id
        self.name = name
        self.concept = concept

    def json(self) -> Dict:
        """Return the JSON representation of the concept set."""
        return {
            "id": self.id,
            "name": self.name,
            "expression": {
                "items": [{"concept": self.concept, "includeDescendents": True}]
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
        self._concept_sets: List[ConceptSet] = []

    def get(self, name: str, concept: Dict) -> Optional[ConceptSet]:
        """
        Get a concept set from the manager.

        Args:
          name: The name of the concept set.
          concept: The concept in the concept set.

        Returns:
          The concept set.
        """

        concept_set = ConceptSet(-1, name, concept)

        if concept_set in self._concept_sets:
            return self._concept_sets[self._concept_sets.index(concept_set)]

        return None

    def add(self, name: str, concept: Dict) -> ConceptSet:
        """
        Add a concept set to the manager.

        Args:
          name: The name of the concept set.
          concept: The concept in the concept set.

        Returns:
          The concept set.
        """

        concept_set = ConceptSet(len(self._concept_sets), name, concept)

        if concept_set in self._concept_sets:
            return self._concept_sets[self._concept_sets.index(concept_set)]

        self._concept_sets.append(concept_set)

        return concept_set

    def json(self) -> List[Dict]:
        """
        Return the JSON representation of the concept sets.
        """
        return [concept_set.json() for concept_set in self._concept_sets]
