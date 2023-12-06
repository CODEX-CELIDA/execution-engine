import multiprocessing.shared_memory
from enum import Enum, auto

import pandas as pd
import sympy
from sympy import And, Not, Or

from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.sql import OMOPSQLClient
from execution_engine.settings import config
from execution_engine.task import process


def get_engine() -> OMOPSQLClient:
    """
    Returns a OMOPSQLClient object.
    """
    return OMOPSQLClient(
        **config.omop.dict(by_alias=True), timezone=config.celida_ee_timezone
    )


class TaskState(Enum):
    """
    An enum representing the state of a Task object.
    """

    NOT_RUN = auto()
    RUNNING = auto()
    FINISHED = auto()


class Task:
    """
    A Task object represents a task that needs to be run.
    """

    def __init__(
        self, expr: sympy.Expr, criterion: Criterion, usage_count: int
    ) -> None:
        self.expr = expr
        self.criterion = criterion
        self.usage_count = usage_count
        self.state = TaskState.NOT_RUN
        self.dependencies: list[Task] = []

    def run(self, data: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Runs the task.
        """
        by = ["person_id", "concept_id"]

        if self.state == TaskState.NOT_RUN:
            self.state = TaskState.RUNNING
            try:
                if len(self.dependencies) == 0:
                    # query = self.criterion.create_query()
                    # result = get_engine().query(query)

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
                elif len(self.dependencies) == 1:
                    assert self.expr.is_Not, "Dependency is not a Not expression."
                    result = process.invert(data[0])

                elif len(self.dependencies) >= 2:
                    if isinstance(self.expr, And):
                        result = process.merge_intervals(data, by)
                    elif isinstance(self.expr, Or):
                        result = process.intersect_intervals(data, by)
                    else:
                        raise ValueError(f"Unsupported expression type: {self.expr}")

                self.state = TaskState.FINISHED
            except Exception as e:
                self.state = TaskState.NOT_RUN
                print(f"Task failed with error: {e}")
            self.usage_count -= 1
        elif self.usage_count > 0 and self.state in [
            TaskState.RUNNING,
            TaskState.FINISHED,
        ]:
            self.usage_count -= 1

        return result

    def name(self) -> str:
        """
        Returns the name of the Task object.
        """
        return self.criterion.name

    def is_finished(self) -> bool:
        """
        Returns True if the task is finished, False otherwise.
        """
        return self.state == TaskState.FINISHED

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object.
        """
        return f"Task({self.expr}, criterion={self.criterion}, usage_count={self.usage_count}): state={self.state}"


def create_tasks_and_dependencies(
    expr: sympy.Expr, task_mapping: dict, criterion_hashmap: dict, usage_counts: dict
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
    usage_counts : dict
        A mapping of expressions to usage counts.

    Returns
    -------
    Task
        The (root) Task object for the expression.
    """
    if expr in task_mapping:
        return task_mapping[expr]

    current_criterion = criterion_hashmap.get(expr, None)
    current_task = Task(
        expr, current_criterion, usage_count=usage_counts[expr]
    )  # Create a Task object for the current expression

    # Dependencies are the children in the expression tree
    dependencies = []
    if isinstance(expr, (And, Or, Not)):
        for arg in expr.args:
            child_task = create_tasks_and_dependencies(
                arg, task_mapping, criterion_hashmap, usage_counts
            )
            dependencies.append(child_task)

    current_task.dependencies = dependencies
    task_mapping[expr] = current_task

    return current_task
