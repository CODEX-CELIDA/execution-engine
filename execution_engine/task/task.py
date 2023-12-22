import logging
from enum import Enum, auto

import pandas as pd

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida.tables import RecommendationResultInterval
from execution_engine.omop.sqlclient import OMOPSQLClient
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

    def get_base_task(self) -> "Task":
        """
        Returns the base task of the task.
        """

        def find_base_task(task: Task) -> Task:
            """
            Recursively find the base task for a given task.

            :param task: The Task object to find the base task for.
            :return: The base Task object with no dependencies.
            """
            if not task.dependencies:  # No dependencies means this is the base task
                return task
            else:
                # Recursively find the base task for the first dependency.
                # This assumes that all dependencies eventually lead to the same base task.
                return find_base_task(task.dependencies[0])

        return find_base_task(self)

    def run(
        self, data: list[pd.DataFrame], base_data: pd.DataFrame | None, params: dict
    ) -> pd.DataFrame:
        """
        Runs the task.

        :param data: The input data.
        :param base_data: The result of the base criterion or None, if this is the base criterion.
        :param params: The parameters.
        :return: The result of the task.
        """

        # todo: should we only use the params from the task instead of the parameter?
        params = params | self.params

        observation_window = TimeRange(
            start=params["observation_start_datetime"],
            end=params["observation_end_datetime"],
            name="observation_window",
        )

        self.status = TaskStatus.RUNNING
        logging.info(f"Running task '{self.name()}'")

        try:
            if len(self.dependencies) == 0 or self.expr.is_Atom:
                # atomic expressions (i.e. criterion)
                result = self.handle_criterion(params, observation_window)

                self.store_result_in_db(result, params)

                # here, already combine intervals with type NO_DATA and POSITIVE and drop NEGATIVE
                result = self.consolidate_intervals(result)

            else:
                # non-atomic expressions (i.e. logical operations on criteria)
                if len(self.dependencies) == 1:
                    # only Not has one dependency
                    result = self.handle_unary_logical_operator(
                        data, base_data, observation_window
                    )
                elif len(self.dependencies) >= 2:
                    # And and Or have two or more dependencies
                    result = self.handle_binary_logical_operator(data)

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

    def handle_criterion(
        self, params: dict, observation_window: TimeRange
    ) -> pd.DataFrame:
        """
        Handles a criterion by querying the database.

        :observation_window: The observation window.
        :return: A DataFrame with the result of the query.
        """
        engine = get_engine()
        query = self.criterion.create_query()
        result = engine.query(query, **params)

        result = self.criterion.process_result(result, observation_window)

        return result

    def consolidate_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidates the intervals of the DataFrame by dropping NEGATIVE intervals and considering
        all remaining intervals (i.e. NO_DATA) as POSITIVE.

        :param df: The DataFrame with the intervals.
        :return: A DataFrame with the consolidated intervals.
        """
        df = df[df["interval_type"] != IntervalType.NEGATIVE]
        df["interval_type"] = IntervalType.POSITIVE

        # merge overlapping/adjacent intervals to reduce the number of intervals
        return process.merge_intervals([df], self.by)

    def handle_unary_logical_operator(
        self,
        data: list[pd.DataFrame],
        base_data: pd.DataFrame,
        observation_window: TimeRange,
    ) -> pd.DataFrame:
        """
        Handles a unary logical operator (Not) by inverting the intervals of the dependency.

        Patients for which no data is available in the data are identified from the result of the base criterion
        and the full observation window is added for these patients. Otherwise, these patients would be missing
        from the result.

        :param data: The input data.
        :param base_data: The result of the base criterion (required to add full observation window for patients with no data).
        :param observation_window: The observation window.
        :return: A DataFrame with the inverted intervals.
        """
        assert self.expr.is_Not, "Dependency is not a Not expression."

        result = process.invert_intervals(data[0], self.by, observation_window)

        # need to add all persons that are not in the result
        result = process.insert_missing_intervals(
            result, base_data, self.by, observation_window
        )

        return result

    def handle_binary_logical_operator(self, data: list[pd.DataFrame]) -> pd.DataFrame:
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
            pi_pair_id=params["pi_pair_id"],
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
