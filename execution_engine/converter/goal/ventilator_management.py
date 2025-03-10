from typing import Type

from fhir.resources.plandefinition import PlanDefinitionGoal

from execution_engine.constants import SCT_VENTILATOR_CARE_AND_ADJUSTMENT
from execution_engine.converter.criterion import parse_code, parse_value
from execution_engine.converter.goal.abstract import Goal
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.custom.tidal_volume import (
    TidalVolumePerIdealBodyWeight,
)
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.vocabulary import CODEXCELIDA, SNOMEDCT
from execution_engine.util import logic
from execution_engine.util.value import Value

CUSTOM_GOALS: dict[Concept, Type] = {
    CODEXCELIDA.map["tvpibw"]: TidalVolumePerIdealBodyWeight,
}


class VentilatorManagementGoal(Goal):
    """
    A ventilator management goal.
    """

    _concept_vocabulary = SNOMEDCT
    _concept_code = SCT_VENTILATOR_CARE_AND_ADJUSTMENT

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
        self._criterion_class = Measurement

    @classmethod
    def from_fhir(cls, goal: PlanDefinitionGoal) -> "VentilatorManagementGoal":
        """
        Converts a FHIR goal to a ventilator management goal.
        """
        if len(goal.target) != 1:
            raise NotImplementedError("Only one target is supported")

        target = goal.target[0]

        value = parse_value(target, value_prefix="detail")
        code = parse_code(target.measure)

        return cls(exclude=False, code=code, value=value)

    def to_positive_expression(self) -> logic.Symbol:
        """
        Converts the goal to a criterion.
        """
        if self._code in CUSTOM_GOALS:
            cls = CUSTOM_GOALS[self._code]
            return logic.Symbol(
                cls(
                    concept=self._code,
                    value=self._value,
                )
            )

        return logic.Symbol(
            Measurement(
                concept=self._code,
                value=self._value,
            )
        )
