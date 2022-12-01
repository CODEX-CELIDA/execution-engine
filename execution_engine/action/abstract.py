from abc import ABC, abstractmethod
from typing import Type

from ..fhir.recommendation import Recommendation
from ..fhir.util import get_coding
from ..goal import AbstractGoal
from ..omop.criterion.abstract import Criterion
from ..omop.vocabulary import AbstractVocabulary


class AbstractAction(ABC):
    """
    An abstract action.

    An instance of this class represents an action entry of the PlanDefinition resource in the context of
    CPG-on-EBM-on-FHIR. In the Implementation Guide (specifically, the InterventionPlan profile),
    several types of actions are defined, including:
    - Drug Administration
    - Ventilator Management
    - Body Positioning
    Each of these slices from the Implementation Guide is represented by a subclass of this class.

    Subclasses must define the following methods:
    - valid: returns True if the supplied action  falls within the scope of the subclass
    - to_omop: converts the action to an OMOP criterion
    - from_fhir: creates a new instance of the subclass from a FHIR PlanDefinition.action element

    """

    _criterion_class: Type[Criterion]
    _concept_code: str
    _concept_vocabulary: Type[AbstractVocabulary]
    _goal_required: bool
    _goal_class: Type[AbstractGoal] | None
    _goal: AbstractGoal | None

    @classmethod
    @abstractmethod
    def from_fhir(cls, action: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()

    @classmethod
    def valid(
        cls,
        action_def: Recommendation.Action,
    ) -> bool:
        """Checks if the given FHIR definition is a valid action in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(action_def.action.code)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )

    @abstractmethod
    def to_criterion(self) -> Criterion:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()
