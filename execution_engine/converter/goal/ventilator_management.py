from fhir.resources.plandefinition import PlanDefinitionGoal

from ...constants import SCT_VENTILATOR_CARE_AND_ADJUSTMENT
from ...omop.criterion.abstract import Criterion
from ...omop.vocabulary import SNOMEDCT
from .abstract import Goal


class VentilatorManagement(Goal):
    """
    A ventilator management goal.
    """

    _concept_vocabulary = SNOMEDCT
    _concept_code = SCT_VENTILATOR_CARE_AND_ADJUSTMENT

    @classmethod
    def from_fhir(cls, goal: PlanDefinitionGoal) -> "VentilatorManagement":
        """
        Converts a FHIR goal to a ventilator management goal.
        """
        pass

    def to_criterion(self) -> Criterion:
        """
        Converts the goal to a criterion.
        """
        pass
