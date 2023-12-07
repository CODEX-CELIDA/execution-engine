from enum import Enum, auto

import pandas as pd

import execution_engine.util.cohort_logic as logic
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

    def __init__(self, expr: logic.Expr, criterion: Criterion) -> None:
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
            if len(self.dependencies) == 0 or self.expr.is_Atom:
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
            start=params["observation_window_start"],
            end=params["observation_window_end"],
            name="observation_window",
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
        if isinstance(self.expr, logic.And):
            result = process.merge_intervals(data, by)
        elif isinstance(self.expr, logic.Or):
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
