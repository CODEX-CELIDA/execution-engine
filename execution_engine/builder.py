from typing import TYPE_CHECKING, TypedDict

from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.action.assessment import AssessmentAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.procedure import ProcedureAction
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
from execution_engine.converter.criterion import CriterionConverter
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.goal.assessment_scale import AssessmentScaleGoal
from execution_engine.converter.goal.laboratory_value import LaboratoryValueGoal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)
from execution_engine.converter.time_from_event.abstract import TemporalIndicator

if TYPE_CHECKING:
    from execution_engine.execution_engine import ExecutionEngine


class CriterionConverterType(TypedDict):
    """
    A dictionary type that maps criterion types to a list of CriterionConverter types.
    """

    characteristic: list[type[CriterionConverter]]
    action: list[type[CriterionConverter]]
    goal: list[type[CriterionConverter]]
    time_from_event: list[type[TemporalIndicator]]


_default_converters: CriterionConverterType = {
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
        ProcedureAction,
    ],
    "goal": [LaboratoryValueGoal, VentilatorManagementGoal, AssessmentScaleGoal],
    "time_from_event": [],
}


def default_execution_engine_builder() -> "ExecutionEngineBuilder":
    """
    Creates a default ExecutionEngineBuilder with default converters.
    """
    builder = ExecutionEngineBuilder()

    builder.set_characteristic_converters(_default_converters["characteristic"])
    builder.set_action_converters(_default_converters["action"])
    builder.set_goal_converters(_default_converters["goal"])
    builder.set_time_from_event_converters(_default_converters["time_from_event"])

    return builder


class ExecutionEngineBuilder:
    """
    A builder for ExecutionEngine instances.

    This builder allows for the specification of characteristic, action, goal, and timeFromEvent converters to be
    used by the ExecutionEngine.
    """

    def __init__(self) -> None:
        self.characteristic_converters: list[type[CriterionConverter]] = []
        self.action_converters: list[type[CriterionConverter]] = []
        self.goal_converters: list[type[CriterionConverter]] = []
        self.time_from_event_converters: list[type[TemporalIndicator]] = []

    def set_characteristic_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets (overwrites) the characteristic converters for this builder.
        """
        self.characteristic_converters.clear()
        for converter_type in converters:
            self.append_characteristic_converter(converter_type)
        return self

    def set_action_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets (overwrites) the action converters for this builder.
        """
        self.action_converters.clear()
        for converter_type in converters:
            self.append_action_converter(converter_type)
        return self

    def set_goal_converters(
        self, converters: list[type[CriterionConverter]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets (overwrites) the goal converters for this builder.
        """
        self.goal_converters.clear()
        for converter_type in converters:
            self.append_goal_converter(converter_type)
        return self

    def set_time_from_event_converters(
        self, converters: list[type[TemporalIndicator]]
    ) -> "ExecutionEngineBuilder":
        """
        Sets (overwrites) the time from event converters for this builder.
        """
        self.time_from_event_converters.clear()

        for converter_type in converters:
            self.append_time_from_event_converter(converter_type)

        return self

    def append_characteristic_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Appends a single characteristic converter at the end of the list.
        """
        if not issubclass(converter_type, AbstractCharacteristic):
            raise ValueError(f"Invalid Characteristic converter type: {converter_type}")
        self.characteristic_converters.append(converter_type)
        return self

    def prepend_characteristic_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Inserts a single characteristic converter at the front of the list.
        """
        if not issubclass(converter_type, AbstractCharacteristic):
            raise ValueError(f"Invalid Characteristic converter type: {converter_type}")
        self.characteristic_converters.insert(0, converter_type)
        return self

    def append_action_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Appends a single action converter at the end of the list.
        """
        if not issubclass(converter_type, AbstractAction):
            raise ValueError(f"Invalid Action converter type: {converter_type}")
        self.action_converters.append(converter_type)
        return self

    def prepend_action_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Inserts a single action converter at the front of the list.
        """
        if not issubclass(converter_type, AbstractAction):
            raise ValueError(f"Invalid Action converter type: {converter_type}")
        self.action_converters.insert(0, converter_type)
        return self

    def append_goal_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Appends a single goal converter at the end of the list.
        """
        if not issubclass(converter_type, Goal):
            raise ValueError(f"Invalid Goal converter type: {converter_type}")
        self.goal_converters.append(converter_type)
        return self

    def prepend_goal_converter(
        self, converter_type: type[CriterionConverter]
    ) -> "ExecutionEngineBuilder":
        """
        Inserts a single goal converter at the front of the list.
        """
        if not issubclass(converter_type, Goal):
            raise ValueError(f"Invalid Goal converter type: {converter_type}")
        self.goal_converters.insert(0, converter_type)
        return self

    def append_time_from_event_converter(
        self, converter_type: type[TemporalIndicator]
    ) -> "ExecutionEngineBuilder":
        """
        Appends a single time_from_event converter at the end of the list.
        """
        if not issubclass(converter_type, TemporalIndicator):
            raise ValueError(f"Invalid TimeFromEvent converter type: {converter_type}")
        self.time_from_event_converters.append(converter_type)
        return self

    def prepend_time_from_event_converter(
        self, converter_type: type[TemporalIndicator]
    ) -> "ExecutionEngineBuilder":
        """
        Inserts a single time_from_event converter at the front of the list.
        """
        if not issubclass(converter_type, TemporalIndicator):
            raise ValueError(f"Invalid TimeFromEvent converter type: {converter_type}")
        self.time_from_event_converters.insert(0, converter_type)
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
