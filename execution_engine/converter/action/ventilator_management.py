from ...fhir.recommendation import Recommendation
from ...omop.vocabulary import SNOMEDCT
from ..goal.ventilator_management import VentilatorManagement
from .abstract import AbstractAction


class VentilatorManagementAction(AbstractAction):
    """
    A ventilator management action.
    """

    _concept_code = "410210009"  # Ventilator care management (procedure)
    _concept_vocabulary = SNOMEDCT
    _goal_type = VentilatorManagement  # todo: Most likely there is no 1:1 relationship between action and goal types
    _goal_required = True

    @classmethod
    def from_fhir(cls, action: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()
