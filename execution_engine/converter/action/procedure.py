from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.criterion import parse_code
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.observation import Observation
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.vocabulary import SNOMEDCT
from execution_engine.util import logic
from execution_engine.util.types import Timing


class ProcedureAction(AbstractAction):
    """
    An ProcedureAction is an action that describes a procedure to be performed

    This action tests whether the procedure has been performed by determining whether it is
    is present in the respective OMOP CDM table.
    """

    _concept_code = "71388002"  # Procedure (procedure)
    _concept_vocabulary = SNOMEDCT

    def __init__(
        self,
        exclude: bool,
        code: Concept,
        timing: Timing | None = None,
    ) -> None:
        """
        Initialize the procedure action.
        """
        super().__init__(exclude=exclude)
        self._code = code
        self._timing = timing

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> AbstractAction:
        """Creates a new action from a FHIR PlanDefinition."""
        assert (
            action_def.activity_definition_fhir is not None
        ), f"ActivityDefinition is required for {cls.__class__.__name__}"
        assert (
            action_def.activity_definition_fhir.code is not None
        ), f"Code is required for {cls.__class__.__name__}"
        assert (
            action_def.activity_definition_fhir.timingTiming is not None
        ), f"Timing is required for {cls.__class__.__name__}"

        code = parse_code(action_def.activity_definition_fhir.code)
        timing = cls.process_timing(action_def.activity_definition_fhir.timingTiming)

        exclude = action_def.activity_definition_fhir.doNotPerform

        return cls(exclude=exclude, code=code, timing=timing)

    def _to_expression(self) -> logic.Symbol:
        """Converts this characteristic to a Criterion."""

        criterion: Criterion

        match self._code.domain_id:
            case "Procedure":
                criterion = ProcedureOccurrence(
                    concept=self._code,
                    timing=self._timing,
                )
            case "Measurement":
                # we need to explicitly set the VALUE_REQUIRED flag to false, otherwise creating the query will raise an error
                # as Observation and Measurement normally expect a value.
                criterion = Measurement(
                    concept=self._code,
                    override_value_required=False,
                    timing=self._timing,
                )
            case "Observation":
                # we need to explicitly set the VALUE_REQUIRED flag to false, otherwise creating the query will raise an error
                # as Observation and Measurement normally expect a value.
                criterion = Observation(
                    concept=self._code,
                    override_value_required=False,
                    timing=self._timing,
                )
            case _:
                raise ValueError(
                    f"Concept domain {self._code.domain_id} is not supported for {self.__class__.__name__}]"
                )

        return logic.Symbol(criterion=criterion)
