from ...fhir.recommendation import Recommendation
from ...omop.criterion.abstract import Criterion
from ...omop.criterion.combination import CriterionCombination
from ...omop.vocabulary import SNOMEDCT, VocabularyNotFoundError
from ..converter import parse_code
from ..goal.ventilator_management import VentilatorManagementGoal
from .abstract import AbstractAction


class VentilatorManagementAction(AbstractAction):
    """
    A ventilator management action.
    """

    _concept_code = "410210009"  # Ventilator care management (procedure)
    _concept_vocabulary = SNOMEDCT
    _goal_type = VentilatorManagementGoal  # todo: Most likely there is no 1:1 relationship between action and goal types
    _goal_required = True

    @classmethod
    def from_fhir(cls, action_def: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        if action_def.activity is not None:
            raise ValueError("VentilatorManagementAction does not support activity")

        assert action_def.goals, "VentilatorManagementAction must have goals"

        # only using first goal for name
        goal = action_def.goals[0]
        code = parse_code(goal.target[0].measure)

        return cls(
            name=code.name, exclude=False
        )  # fixme: no way to exclude goals (e.g. "do not ventilate")

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""
        return None
