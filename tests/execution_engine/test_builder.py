from unittest.mock import Mock

from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.converter.converter import CriterionConverter
from execution_engine.converter.parser.factory import FhirRecommendationParserFactory
from execution_engine.execution_engine import ExecutionEngine


class TestExecutionEngineBuilder:
    def setup_method(self):
        self.builder = ExecutionEngineBuilder()
        self.mock_converter = Mock(spec=CriterionConverter)

    def test_set_characteristic_converters(self):
        """Test setting characteristic converters."""
        converters = [self.mock_converter, self.mock_converter]
        result = self.builder.set_characteristic_converters(converters)
        assert result == self.builder
        assert self.builder.characteristic_converters == converters

    def test_set_action_converters(self):
        """Test setting action converters."""
        converters = [self.mock_converter, self.mock_converter]
        result = self.builder.set_action_converters(converters)
        assert result == self.builder
        assert self.builder.action_converters == converters

    def test_set_goal_converters(self):
        """Test setting goal converters."""
        converters = [self.mock_converter, self.mock_converter]
        result = self.builder.set_goal_converters(converters)
        assert result == self.builder
        assert self.builder.goal_converters == converters

    def test_build_execution_engine(self):
        """Test building ExecutionEngine with the set converters."""
        self.builder.set_characteristic_converters([self.mock_converter])
        self.builder.set_action_converters([self.mock_converter])
        self.builder.set_goal_converters([self.mock_converter])
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
