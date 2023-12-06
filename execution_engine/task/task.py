from enum import Enum, auto

import pandas as pd
import sympy
from sympy import And, Not, Or

from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.sql import OMOPSQLClient
from execution_engine.settings import config
from execution_engine.task import process
from execution_engine.util import TimeRange


def get_engine() -> OMOPSQLClient:
    """
    Returns a OMOPSQLClient object.
    """
    return OMOPSQLClient(
        **config.omop.dict(by_alias=True), timezone=config.celida_ee_timezone
    )


class TaskStatus(Enum):
    """
    An enumeration of task statuses.
    """

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class TaskError(Exception):
    """
    A TaskError is raised when a task fails.
    """


class Task:
    """
    A Task object represents a task that needs to be run.
    """

    def __init__(self, expr: sympy.Expr, criterion: Criterion) -> None:
        self.expr = expr
        self.criterion = criterion
        self.dependencies: list[Task] = []
        self.status = TaskStatus.PENDING

    def run(self, data: list[pd.DataFrame], params: dict) -> pd.DataFrame:
        """
        Runs the task.

        :param data: The input data.
        :param params: The parameters.
        :return: The result of the task.
        """

        self.status = TaskStatus.RUNNING

        try:
            if len(self.dependencies) == 0:
                # only criteria have no dependencies, everything else is a combination (or Not)
                result = self.handle_criterion(params)
            elif len(self.dependencies) == 1:
                # only Not has one dependency
                result = self.handle_unary_logical_operator(data, params)
            elif len(self.dependencies) >= 2:
                # And and Or have two or more dependencies
                result = self.handle_binary_logical_operator(data, params)

        except Exception as e:
            self.status = TaskStatus.FAILED
            raise TaskError(f"Task '{self.name()}' failed with error: {e}")

        self.status = TaskStatus.COMPLETED

        return result

    def handle_criterion(self, params: dict) -> pd.DataFrame:
        """
        Handles a criterion by querying the database.

        :param params: The parameters.
        :return: A DataFrame with the result of the query.
        """
        # engine = get_engine()
        # query = self.criterion.create_query()
        # result = engine.query(query)

        # todo: insert result into database

        result = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "concept_id": [1, 1, 1],
                "interval_start": [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                ],
                "interval_end": [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                ],
            }
        )

        return result

    def handle_unary_logical_operator(
        self, data: list[pd.DataFrame], params: dict
    ) -> pd.DataFrame:
        """
        Handles a unary logical operator (Not) by inverting the intervals of the dependency.

        :param data: The input data.
        :param params: The parameters.
        :return: A DataFrame with the inverted intervals.
        """
        assert self.expr.is_Not, "Dependency is not a Not expression."
        observation_window = TimeRange(
            params["observation_window_start"],
            params["observation_window_end"],
            "observation_window",
        )
        result = process.invert(data[0], observation_window)
        return result

    def handle_binary_logical_operator(
        self, data: list[pd.DataFrame], params: dict
    ) -> pd.DataFrame:
        """
        Handles a binary logical operator (And or Or) by merging or intersecting the intervals of the

        :param data: The input data.
        :param params: The parameters.
        :return: A DataFrame with the merged or intersected intervals.
        """
        by = ["person_id", "concept_id"]
        if isinstance(self.expr, And):
            result = process.merge_intervals(data, by)
        elif isinstance(self.expr, Or):
            result = process.intersect_intervals(data, by)
        else:
            raise ValueError(f"Unsupported expression type: {self.expr}")
        return result

    def name(self) -> str:
        """
        Returns the name of the Task object.
        """
        return str(self)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object.
        """
        return f"Task({self.expr}, criterion={self.criterion})"


def create_tasks_and_dependencies(
    expr: sympy.Expr, task_mapping: dict, criterion_hashmap: dict
) -> Task:
    """
    Creates a Task object for an expression and its dependencies.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to create a Task object for.
    task_mapping : dict
        A mapping of expressions to Task objects.
    criterion_hashmap : dict
        A mapping of expressions to Criterion objects.

    Returns
    -------
    Task
        The (root) Task object for the expression.
    """
    if expr in task_mapping:
        return task_mapping[expr]

    current_criterion = criterion_hashmap.get(expr, None)
    current_task = Task(
        expr, current_criterion
    )  # Create a Task object for the current expression

    # Dependencies are the children in the expression tree
    dependencies = []
    if isinstance(expr, (And, Or, Not)):
        for arg in expr.args:
            child_task = create_tasks_and_dependencies(
                arg, task_mapping, criterion_hashmap
            )
            dependencies.append(child_task)

    current_task.dependencies = dependencies
    task_mapping[expr] = current_task

    return current_task
