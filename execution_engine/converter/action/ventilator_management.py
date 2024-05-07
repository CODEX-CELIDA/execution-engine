from typing import Self

from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.vocabulary import SNOMEDCT


class VentilatorManagementAction(AbstractAction):
    """
    A ventilator management action.
    """

    _concept_code = "410210009"  # Ventilator care management (procedure)
    _concept_vocabulary = SNOMEDCT

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> Self:
        """Creates a new action from a FHIR PlanDefinition."""
        if action_def.activity_definition_fhir is not None:
            raise ValueError("VentilatorManagementAction does not support activity")

        assert action_def.goals_fhir, "VentilatorManagementAction must have goals"

        return cls(
            exclude=False,
        )  # fixme: no way to exclude goals (e.g. "do not ventilate")

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""
        return None
