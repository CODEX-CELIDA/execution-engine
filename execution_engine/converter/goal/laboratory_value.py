from fhir.resources.plandefinition import PlanDefinitionGoal

from ...constants import SCT_LAB_FINDINGS_SURVEILLANCE
from ...omop.criterion.abstract import Criterion
from ...omop.vocabulary import SNOMEDCT
from .abstract import Goal


class LaboratoryValue(Goal):
    """
    A goal defining to achieve a certain laboratory value.
    """

    _concept_vocabulary = SNOMEDCT
    _concept_code = SCT_LAB_FINDINGS_SURVEILLANCE

    @classmethod
    def from_fhir(cls, goal: PlanDefinitionGoal) -> "LaboratoryValue":
        """
        Converts a FHIR goal to a laboratory value goal.
        """
        pass

    def to_criterion(self) -> Criterion:
        """
        Converts the goal to a criterion.
        """
        pass
