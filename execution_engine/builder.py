from typing import TYPE_CHECKING

from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.action.assessment import AssessmentAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.ventilator_management import (
    VentilatorManagementAction,
)
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.characteristic.allergy import AllergyCharacteristic
from execution_engine.converter.characteristic.condition import ConditionCharacteristic
from execution_engine.converter.characteristic.episode_of_care import (
    EpisodeOfCareCharacteristic,
)
from execution_engine.converter.characteristic.observation import (
    ObservationCharacteristic,
)
from execution_engine.converter.characteristic.procedure import ProcedureCharacteristic
from execution_engine.converter.characteristic.radiology import RadiologyCharacteristic
from execution_engine.converter.converter import CriterionConverter
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.goal.assessment_scale import AssessmentScaleGoal
from execution_engine.converter.goal.laboratory_value import LaboratoryValueGoal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)

if TYPE_CHECKING:
    from execution_engine.execution_engine import ExecutionEngine

_default_converters: dict[str, list[type[CriterionConverter]]] = {
    "characteristic": [
        ConditionCharacteristic,
        AllergyCharacteristic,
        RadiologyCharacteristic,
        ProcedureCharacteristic,
        EpisodeOfCareCharacteristic,
        ObservationCharacteristic,
    ],
    "action": [
        DrugAdministrationAction,
        VentilatorManagementAction,
        BodyPositioningAction,
        AssessmentAction,
    ],
    "goal": [LaboratoryValueGoal, VentilatorManagementGoal, AssessmentScaleGoal],
}


def default_execution_engine_builder() -> "ExecutionEngineBuilder":
    """
    Creates a default ExecutionEngineBuilder with default converters.
    """
    builder = ExecutionEngineBuilder()
    # Assuming DefaultConverter1, DefaultConverter2, DefaultConverter3 are defined somewhere
    builder.set_characteristic_converters(_default_converters["characteristic"])
    builder.set_action_converters(_default_converters["action"])
    builder.set_goal_converters(_default_converters["goal"])

    return builder


class ExecutionEngineBuilder:
    """
    A builder for ExecutionEngine instances.

    This builder allows for the specification of characteristic, action, and goal converters to be used by the
    ExecutionEngine.
    """

    def __init__(self) -> None:

        self.characteristic_converters: list[type[CriterionConverter]] = []
        self.action_converters: list[type[CriterionConverter]] = []
        self.goal_converters: list[type[CriterionConverter]] = []

    def set_characteristic_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets the characteristic converters for this builder.

        :param converters: The characteristic converters to set.
        :return: The builder instance.
        """
        for converter_type in converters:
            if not issubclass(converter_type, AbstractCharacteristic):
                raise ValueError(
                    f"Invalid Characteristic converter type: {converter_type}"
                )

        self.characteristic_converters = converters
        return self

    def set_action_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets the action converters for this builder.

        :param converters: The action converters to set.
        :return: The builder instance.
        """
        for converter in converters:
            if not issubclass(converter, AbstractAction):
                raise ValueError(f"Invalid Action converter type: {converter}")

        self.action_converters = converters
        return self

    def set_goal_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets the goal converters for this builder.

        :param converters: The goal converters to set.
        :return: The builder instance.
        """
        for converter in converters:
            if not issubclass(converter, Goal):
                raise ValueError(f"Invalid Goal converter type: {converter}")

        self.goal_converters = converters
        return self

    def build(self, verbose: bool = False) -> "ExecutionEngine":
        """
        Builds an ExecutionEngine with the specified converters.

        :param verbose: Whether to print verbose output.
        :return: A new ExecutionEngine instance.
        """
        # prevent circular import
        from execution_engine.execution_engine import ExecutionEngine

        return ExecutionEngine(builder=self, verbose=verbose)
