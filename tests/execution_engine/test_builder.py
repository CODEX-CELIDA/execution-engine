from execution_engine.builder import (
    ExecutionEngineBuilder,
    _default_converters,
    default_execution_engine_builder,
)
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.characteristic.condition import ConditionCharacteristic
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)
from execution_engine.converter.parser.factory import FhirRecommendationParserFactory
from execution_engine.execution_engine import ExecutionEngine


class TestExecutionEngineBuilder:
    def setup_method(self):
        self.builder = ExecutionEngineBuilder()

    def test_set_characteristic_converters(self):
        """Test setting characteristic converters."""
        converters = [AbstractCharacteristic, AbstractCharacteristic]
        result = self.builder.set_characteristic_converters(converters)
        assert result == self.builder
        assert self.builder.characteristic_converters == converters

    def test_set_action_converters(self):
        """Test setting action converters."""
        converters = [AbstractAction, AbstractAction]
        result = self.builder.set_action_converters(converters)
        assert result == self.builder
        assert self.builder.action_converters == converters

    def test_set_goal_converters(self):
        """Test setting goal converters."""
        converters = [Goal, Goal]
        result = self.builder.set_goal_converters(converters)
        assert result == self.builder
        assert self.builder.goal_converters == converters

    def test_build_execution_engine(self):
        """Test building ExecutionEngine with the set converters."""
        self.builder.set_characteristic_converters([ConditionCharacteristic])
        self.builder.set_action_converters([DrugAdministrationAction])
        self.builder.set_goal_converters([VentilatorManagementGoal])
        engine = self.builder.build(verbose=True)
        assert isinstance(engine, ExecutionEngine)
        assert engine.fhir_parser.builder == self.builder

        # Test that the converters are registered in the FhirRecommendationParserFactory
        fac = FhirRecommendationParserFactory(builder=engine.fhir_parser.builder)

        assert all(
            conv in fac.goal_converters._converters
            for conv in self.builder.goal_converters
        )
        assert all(
            conv in fac.action_converters._converters
            for conv in self.builder.action_converters
        )
        assert all(
            conv in fac.characteristic_converters._converters
            for conv in self.builder.characteristic_converters
        )


class TestDefaultExecutionEngineBuilder:

    def test_builder_initialization(self):
        """
        Test if ExecutionEngineBuilder is properly initialized.
        """
        builder = default_execution_engine_builder()
        assert isinstance(
            builder, ExecutionEngineBuilder
        ), "Builder is not an instance of ExecutionEngineBuilder"

    def test_characteristic_converters_set_correctly(self):
        """
        Test if characteristic converters are set as expected.
        """
        builder = default_execution_engine_builder()
        assert (
            builder.characteristic_converters == _default_converters["characteristic"]
        ), "Characteristic converters are not set correctly"

    def test_action_converters_set_correctly(self):
        """
        Test if action converters are set as expected.
        """
        builder = default_execution_engine_builder()
        assert (
            builder.action_converters == _default_converters["action"]
        ), "Action converters are not set correctly"

    def test_goal_converters_set_correctly(self):
        """
        Test if goal converters are set as expected.
        """
        builder = default_execution_engine_builder()
        assert (
            builder.goal_converters == _default_converters["goal"]
        ), "Goal converters are not set correctly"
