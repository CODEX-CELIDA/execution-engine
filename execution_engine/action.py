import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Type

from fhir.resources.plandefinition import PlanDefinitionAction, PlanDefinitionGoal

from .fhir.util import get_coding
from .goal import AbstractGoal, LaboratoryValueGoal, VentilatorManagementGoal
from .omop.concepts import ConceptSet
from .omop.criterion import Criterion
from .omop.vocabulary import SNOMEDCT, AbstractVocabulary


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
    def from_fhir(cls, action: PlanDefinitionAction) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()

    @classmethod
    def valid(
        cls,
        action_definition: PlanDefinitionAction,
    ) -> bool:
        """Checks if the given FHIR definition is a valid action in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(action_definition.code)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )

    @abstractmethod
    def to_omop(self) -> Tuple[ConceptSet, Criterion]:
        """Convert this actions to an OMOP criterion."""
        raise NotImplementedError()


class ActionFactory:
    """Factory for creating Action objects from PlanDefinition.action."""

    def __init__(self) -> None:
        self._action_types: list[Type[AbstractAction]] = []

    def register_action_type(self, action: Type[AbstractAction]) -> None:
        """Register a new action type."""
        self._action_types.append(action)

    def get_action(
        self, fhir: PlanDefinitionAction, goals: list[PlanDefinitionGoal]
    ) -> AbstractAction:
        """Get the action type for the given CodeableConcept from PlanDefinition.action.code."""
        for action in self._action_types:
            if action.valid(fhir):
                return action.from_fhir(fhir)
        raise ValueError("No action type matched the FHIR definition.")


class DrugAdministrationAction(AbstractAction):
    """
    A drug administration action.
    """

    _concept_code = "432102000"  # Administration of substance (procedure)
    _concept_vocabulary = SNOMEDCT
    _goal_type = LaboratoryValueGoal  # todo: Most likely there is no 1:1 relationship between action and goal types

    @classmethod
    def from_fhir(cls, action: PlanDefinitionAction) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()


class VentilatorManagementAction(AbstractAction):
    """
    A ventilator management action.
    """

    _concept_code = "410210009"  # Ventilator care management (procedure)
    _concept_vocabulary = SNOMEDCT
    _goal_type = VentilatorManagementGoal  # todo: Most likely there is no 1:1 relationship between action and goal types
    _goal_required = True

    @classmethod
    def from_fhir(cls, action: PlanDefinitionAction) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()


class BodyPositioningAction(AbstractAction):
    """
    A body positioning action.
    """

    _concept_code = "229824005"  # Positioning patient (procedure)
    _concept_vocabulary = SNOMEDCT

    @classmethod
    def from_fhir(cls, action: PlanDefinitionAction) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()


class ActionSelectionBehavior:
    """Mapping from FHIR PlanDefinition.action.selectionBehavior to OMOP InclusionRule Type/Count."""

    _map = {
        "any": "ANY",
        "all": "ALL",
        "all-or-none": "ALL_OR_NONE",
        "exactly-one": "EXACTLY_ONE",
        "at-most-one": "AT_MOST_ONE",
        "one-or-more": "ONE_OR_MORE",
    }

    def __init__(self, behavior: str) -> None:
        if behavior not in self._map:
            raise ValueError(f"Invalid action selection behavior: {behavior}")
        elif self._map[behavior] is None:
            raise ValueError(f"Unsupported action selection behavior: {behavior}")

        self._behavior = behavior
