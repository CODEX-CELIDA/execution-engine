from fhir.resources.plandefinition import PlanDefinitionGoal

from execution_engine.constants import SCT_LAB_FINDINGS_SURVEILLANCE
from execution_engine.converter.criterion import parse_code_value
from execution_engine.converter.goal.abstract import Goal
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.vocabulary import SNOMEDCT
from execution_engine.util.value import Value


class LaboratoryValueGoal(Goal):
    """
    A goal defining to achieve a certain laboratory value.
    """

    _concept_vocabulary = SNOMEDCT
    _concept_code = SCT_LAB_FINDINGS_SURVEILLANCE

    def __init__(
        self,
        exclude: bool,
        code: Concept,
        value: Value,
    ) -> None:
        """
        Initialize the goal.
        """
        super().__init__(exclude=exclude)
        self._code = code
        self._value = value

    @classmethod
    def from_fhir(cls, goal: PlanDefinitionGoal) -> "LaboratoryValueGoal":
        """
        Converts a FHIR goal to a laboratory value goal.
        """

        if len(goal.target) != 1:
            raise NotImplementedError("Only one target is supported")

        target = goal.target[0]

        code, value = parse_code_value(target.measure, target, value_prefix="detail")

        return cls(exclude=False, code=code, value=value)

    def to_positive_criterion(self) -> Criterion:
        """
        Converts the goal to a criterion.
        """
        return Measurement(
            concept=self._code,
            value=self._value,
        )
