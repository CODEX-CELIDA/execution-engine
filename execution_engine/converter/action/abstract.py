from abc import abstractmethod
from typing import Type, final

from execution_engine.constants import CohortCategory
from execution_engine.converter.converter import CriterionConverter
from execution_engine.converter.goal.abstract import Goal
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util import AbstractPrivateMethods


class AbstractAction(CriterionConverter, metaclass=AbstractPrivateMethods):
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
    _goals: list[Goal]

    def __init__(self, name: str, exclude: bool):
        super().__init__(name=name, exclude=exclude)
        self._goals = []

    @classmethod
    @abstractmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()

    @classmethod
    def valid(
        cls,
        action_def: RecommendationPlan.Action,
    ) -> bool:
        """Checks if the given FHIR definition is a valid action in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(action_def.action.code)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )

    @abstractmethod
    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()

    @final
    def to_criterion(self) -> Criterion | CriterionCombination:
        """
        Converts this action to a criterion.
        """
        action = self._to_criterion()

        if action is None:
            assert (
                self.goals
            ), "Action without explicit criterion must have at least one goal"

        if self.goals:
            combination = CriterionCombination(
                name=f"{self._name}_plus_goals",
                exclude=self._exclude,  # need to pull up the exclude flag from the criterion into the combination
                category=CohortCategory.INTERVENTION,
                operator=CriterionCombination.Operator("AND"),
            )
            action.exclude = (  # type: ignore   # action is not None, as tested above
                False  # reset the exclude flag, as it is now part of the combination
            )

            combination.add(action)  # type: ignore  # action is not None, as tested above

            for goal in self.goals:
                combination.add(goal.to_criterion())

            return combination

        return action  # type: ignore

    @property
    def goals(self) -> list[Goal]:
        """
        Returns the goals as defined in the FHIR PlanDefinition.
        """
        return self._goals
