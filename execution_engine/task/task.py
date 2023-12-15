import logging
from enum import Enum, auto

import pandas as pd

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida.tables import RecommendationResultInterval
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

    """The columns to group by when merging or intersecting intervals."""
    by = ["person_id", "interval_type"]

    def __init__(
        self,
        expr: logic.Expr,
        criterion: Criterion,
        params: dict | None,
        store_result: bool = False,
    ) -> None:
        self.expr = expr
        self.criterion = criterion
        self.dependencies: list[Task] = []
        self.status = TaskStatus.PENDING
        self.params = params if params is not None else {}
        self.store_result = store_result

    @property
    def category(self) -> CohortCategory:
        """
        Returns the category of the task.
        """
        return self.expr.category

    def run(self, data: list[pd.DataFrame], params: dict) -> pd.DataFrame:
        """
        Runs the task.

        :param data: The input data.
        :param params: The parameters.
        :return: The result of the task.
        """

        # todo: should we only use the params from the task instead of the parameter?
        params = params | self.params

        self.status = TaskStatus.RUNNING
        logging.info(f"Running task '{self.name()}'")

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

            if self.store_result:
                self.store_result_in_db(result, params)

        except Exception as e:
            self.status = TaskStatus.FAILED
            exception_type = type(e).__name__  # Get the type of the exception
            raise TaskError(
                f"Task '{self.name()}' failed with error: {exception_type}('{e}')"
            )

        self.status = TaskStatus.COMPLETED

        return result

    def handle_criterion(self, params: dict) -> pd.DataFrame:
        """
        Handles a criterion by querying the database.

        :param params: The parameters.
        :return: A DataFrame with the result of the query.
        """
        engine = get_engine()
        query = self.criterion.create_query()
        result = engine.query(query, **params)

        # merge overlapping/adjacent intervals to reduce the number of intervals
        result = process.merge_intervals([result], self.by)

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
            start=params["observation_start_datetime"],
            end=params["observation_end_datetime"],
            name="observation_window",
        )
        result = process.invert_intervals(data[0], self.by, observation_window)
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
        if isinstance(self.expr, logic.And):
            result = process.intersect_intervals(data, self.by)
        elif isinstance(self.expr, logic.Or):
            result = process.merge_intervals(data, self.by)
        else:
            raise ValueError(f"Unsupported expression type: {self.expr}")
        return result

    def store_result_in_db(self, result: pd.DataFrame, params: dict) -> None:
        """
        Stores the result in the database.

        :param result: The result to store.
        :param params: The parameters.
        :return: None.
        """
        # todo do we want to assign here (i.e. instead of outside the task somehow or at least as params to the statement)

        if len(result) == 0:
            return

        criterion_id = self.criterion.id if self.criterion is not None else None

        result = result.assign(
            criterion_id=criterion_id,
            plan_id=params["plan_id"],
            recommendation_run_id=params["run_id"],
            cohort_category=self.category,
        )  # todo: can we get category directly instead of storing it in the task?

        with get_engine().begin() as conn:
            conn.execute(
                RecommendationResultInterval.__table__.insert(),
                result.to_dict(orient="records"),
            )

    def name(self) -> str:
        """
        Returns the name of the Task object.
        """
        return str(self)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object.
        """
        if self.expr.is_Atom:
            return f"Task({self.expr}, criterion={self.criterion}, category={self.expr.category})"
        else:
            return f"Task({self.expr}), category={self.expr.category})"
