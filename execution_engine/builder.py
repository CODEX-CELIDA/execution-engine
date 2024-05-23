from typing import TYPE_CHECKING

from execution_engine.converter.converter import CriterionConverter

if TYPE_CHECKING:
    from execution_engine.execution_engine import ExecutionEngine


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
