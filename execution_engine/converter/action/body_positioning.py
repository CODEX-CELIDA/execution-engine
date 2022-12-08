import warnings

from execution_engine.fhir.recommendation import Recommendation
from execution_engine.omop.vocabulary import SNOMEDCT

from ...omop.concepts import Concept
from ...omop.criterion.abstract import Criterion
from ...omop.criterion.combination import CriterionCombination
from ...omop.criterion.procedure_occurrence import ProcedureOccurrence
from ...util import ValueNumber
from ..converter import parse_code, parse_value
from .abstract import AbstractAction


class BodyPositioningAction(AbstractAction):
    """
    A body positioning action.
    """

    _concept_code = "229824005"  # Positioning patient (procedure)
    _concept_vocabulary = SNOMEDCT

    def __init__(
        self,
        name: str,
        exclude: bool,
        code: Concept,
        timing: ValueNumber | None = None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name=name, exclude=exclude)
        self._code = code
        self._timing = timing

    @classmethod
    def from_fhir(cls, action_def: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity is None:
            raise ValueError("BodyPositioningAction must have an activity")

        code = parse_code(action_def.activity.code)
        timing = parse_value(action_def.activity, "timing")

        exclude = action_def.activity.doNotPerform

        return cls(name=code.name, exclude=exclude, code=code, timing=timing)

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""

        return ProcedureOccurrence(
            name=self._name,
            exclude=self._exclude,
            concept=self._code,
            timing=self._timing,
        )
