from abc import ABC, abstractmethod
from typing import Type

from fhir.resources.plandefinition import PlanDefinitionGoal

from execution_engine.converter.criterion import CriterionConverter
from execution_engine.fhir.util import get_coding
from execution_engine.omop.vocabulary import AbstractVocabulary


class Goal(CriterionConverter, ABC):
    """
    PlanDefinition.goal in the context of CPG-on-EBM-on-FHIR.
    """

    _concept_vocabulary: Type[AbstractVocabulary]
    _concept_code: str

    @classmethod
    @abstractmethod
    def from_fhir(cls, goal: PlanDefinitionGoal) -> "CriterionConverter":
        """
        Creates a new goal from a FHIR PlanDefinition.
        """

    @classmethod
    def valid(cls, goal: PlanDefinitionGoal) -> bool:
        """Checks if the given FHIR definition is a valid action in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(goal.category)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )
