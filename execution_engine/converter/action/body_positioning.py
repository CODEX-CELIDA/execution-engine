from typing import Self

from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.criterion import parse_code
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT
from execution_engine.util.types import Timing


class BodyPositioningAction(AbstractAction):
    """
    A body positioning action.
    """

    _concept_code = "229824005"  # Positioning patient (procedure)
    _concept_vocabulary = SNOMEDCT

    def __init__(
        self,
        exclude: bool,
        code: Concept,
        timing: Timing | None = None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(exclude=exclude)
        self._code = code
        self._timing = timing

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> Self:
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity_definition_fhir is None:
            raise ValueError("BodyPositioningAction must have an activity")

        code = parse_code(action_def.activity_definition_fhir.code)
        timing = cls.process_timing(action_def.activity_definition_fhir.timingTiming)

        exclude = action_def.activity_definition_fhir.doNotPerform

        return cls(exclude=exclude, code=code, timing=timing)

    def _to_criterion(self) -> Criterion | LogicalCriterionCombination | None:
        """Converts this characteristic to a Criterion."""

        return ProcedureOccurrence(
            concept=self._code,
            timing=self._timing,
        )
