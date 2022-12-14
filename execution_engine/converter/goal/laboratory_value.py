from fhir.resources.plandefinition import PlanDefinitionGoal

from ...constants import SCT_LAB_FINDINGS_SURVEILLANCE, CohortCategory
from ...omop.concepts import Concept
from ...omop.criterion.abstract import Criterion
from ...omop.criterion.measurement import Measurement
from ...omop.vocabulary import SNOMEDCT
from ...util import Value
from ..converter import parse_code_value
from .abstract import Goal


class LaboratoryValueGoal(Goal):
    """
    A goal defining to achieve a certain laboratory value.
    """

    _concept_vocabulary = SNOMEDCT
    _concept_code = SCT_LAB_FINDINGS_SURVEILLANCE

    def __init__(
        self,
        name: str,
        exclude: bool,
        code: Concept,
        value: Value,
    ) -> None:
        """
        Initialize the goal.
        """
        super().__init__(name=name, exclude=exclude)
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

        return cls(code.name, exclude=False, code=code, value=value)

    def to_criterion(self) -> Criterion:
        """
        Converts the goal to a criterion.
        """
        return Measurement(
            name=self._name,
            exclude=self._exclude,
            category=CohortCategory.INTERVENTION,
            concept=self._code,
            value=self._value,
        )
