from execution_engine.fhir.recommendation import Recommendation
from execution_engine.omop.vocabulary import SNOMEDCT

from .abstract import AbstractAction


class BodyPositioningAction(AbstractAction):
    """
    A body positioning action.
    """

    _concept_code = "229824005"  # Positioning patient (procedure)
    _concept_vocabulary = SNOMEDCT

    @classmethod
    def from_fhir(cls, action: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()
