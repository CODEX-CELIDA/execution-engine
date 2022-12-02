from execution_engine.fhir.recommendation import Recommendation
from execution_engine.omop.vocabulary import SNOMEDCT

from ...omop.criterion.abstract import Criterion
from ...omop.criterion.combination import CriterionCombination
from .abstract import AbstractAction


class BodyPositioningAction(AbstractAction):
    """
    A body positioning action.
    """

    _concept_code = "229824005"  # Positioning patient (procedure)
    _concept_vocabulary = SNOMEDCT

    @classmethod
    def from_fhir(cls, action_def: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()
