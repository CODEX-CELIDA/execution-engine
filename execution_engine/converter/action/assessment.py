from typing import Self, Type

from execution_engine.constants import CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.converter import parse_code
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.observation import Observation
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT
from execution_engine.util.types import Timing


class AssessmentAction(AbstractAction):
    """
    An AssessmentAction is an action that is used to assess a patient's condition.

    This action just tests whether the assessment has been performed by determining whether any value
    is present in the respective OMOP CDM table.
    """

    _concept_code = "386053000"  # Evaluation procedure (procedure)
    _concept_vocabulary = SNOMEDCT

    def __init__(
        self,
        exclude: bool,
        code: Concept,
        timing: Timing | None = None,
    ) -> None:
        """
        Initialize the assessment action.
        """
        super().__init__(exclude=exclude)
        self._code = code
        self._timing = timing

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> Self:
        """Creates a new action from a FHIR PlanDefinition."""
        assert (
            action_def.activity_definition_fhir is not None
        ), "ActivityDefinition is required"
        assert (
            action_def.activity_definition_fhir.code is not None
        ), "Code is required for AssessmentAction"
        assert (
            action_def.activity_definition_fhir.timingTiming is not None
        ), "Timing is required for AssessmentAction"

        code = parse_code(action_def.activity_definition_fhir.code)
        timing = cls.process_timing(action_def.activity_definition_fhir.timingTiming)

        exclude = action_def.activity_definition_fhir.doNotPerform

        return cls(exclude=exclude, code=code, timing=timing)

    def _to_criterion(self) -> Criterion | LogicalCriterionCombination | None:
        """Converts this characteristic to a Criterion."""

        cls: Type[ConceptCriterion]

        match self._code.domain_id:
            case "Procedure":
                cls = ProcedureOccurrence
            case "Measurement":
                cls = Measurement
            case "Observation":
                cls = Observation
            case _:
                raise ValueError(
                    f"Concept domain {self._code.domain_id} is not supported for AssessmentAction"
                )

        criterion = cls(
            category=CohortCategory.INTERVENTION,
            concept=self._code,
            timing=self._timing,
        )

        # we need to explicitly set the VALUE_REQUIRED flag to false after creation of the object,
        # otherwise creating the query will raise an error as Observation and Measurement normally expect
        # a value.
        criterion._OMOP_VALUE_REQUIRED = False

        return criterion
