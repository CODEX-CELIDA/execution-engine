import logging
from enum import Enum, auto

import pandas as pd
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida.tables import RecommendationResultInterval
from execution_engine.omop.sqlclient import OMOPSQLClient
from execution_engine.settings import config
from execution_engine.task import process
from execution_engine.util import TimeRange
from execution_engine.util.interval import IntervalType


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
    # by = ["person_id"]

    def __init__(
        self,
        expr: logic.Expr,
        criterion: Criterion,
        bind_params: dict | None,
        store_result: bool = False,
    ) -> None:
        self.expr = expr
        self.criterion = criterion
        self.dependencies: list[Task] = []
        self.status = TaskStatus.PENDING
        self.bind_params = bind_params if bind_params is not None else {}
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
        self,
        data: list[pd.DataFrame],
        base_data: pd.DataFrame | None,
        bind_params: dict,
    ) -> pd.DataFrame:
        """
        Runs the task.

        :param data: The input data.
        :param base_data: The result of the base criterion or None, if this is the base criterion.
        :param bind_params: The parameters.
        :return: The result of the task.
        """

        # todo: should we only use the params from the task instead of the parameter?
        bind_params = bind_params | self.bind_params

        observation_window = TimeRange(
            start=bind_params["observation_start_datetime"],
            end=bind_params["observation_end_datetime"],
            name="observation_window",
        )

        self.status = TaskStatus.RUNNING
        logging.info(f"Running task '{self.name()}'")

        try:
            if len(self.dependencies) == 0 or self.expr.is_Atom:
                # atomic expressions (i.e. criterion)
                result = self.handle_criterion(
                    bind_params, base_data, observation_window
                )

                self.store_result_in_db(
                    result, base_data, bind_params, observation_window
                )

            else:
                # non-atomic expressions (i.e. logical operations on criteria)
                if isinstance(self.expr, logic.Not):
                    result = self.handle_unary_logical_operator(
                        data, base_data, observation_window
                    )
                elif isinstance(
                    self.expr, (logic.And, logic.Or, logic.NonSimplifiableAnd)
                ):
                    result = self.handle_binary_logical_operator(data)
                elif isinstance(self.expr, logic.LeftDependentToggle):
                    result = self.handle_left_dependent_toggle(
                        left=data[0],
                        right=data[1],
                        base_data=base_data,
                        observation_window=observation_window,
                    )
                elif isinstance(self.expr, logic.NoDataPreservingAnd):
                    result = self.handle_no_data_preserving_operator(
                        data, base_data, observation_window
                    )
                elif isinstance(self.expr, logic.NoDataPreservingOr):
                    result = self.handle_no_data_preserving_operator(
                        data, base_data, observation_window
                    )
                else:
                    raise ValueError(f"Unsupported expression type: {type(self.expr)}")

                if self.store_result:
                    self.store_result_in_db(
                        result, base_data, bind_params, observation_window
                    )

            if set(process.df_dtypes.keys()) != set(result.columns):
                raise TaskError("Invalid result columns.")

        except TaskError as e:  # todo change to exception
            self.status = TaskStatus.FAILED
            exception_type = type(e).__name__  # Get the type of the exception
            raise TaskError(
                f"Task '{self.name()}' failed with error: {exception_type}('{e}')"
            )

        self.status = TaskStatus.COMPLETED

        return result

    def handle_criterion(
        self,
        bind_params: dict,
        base_data: pd.DataFrame | None,
        observation_window: TimeRange,
    ) -> pd.DataFrame:
        """
        Handles a criterion by querying the database.

        :param bind_params: The parameters.
        :param base_data: The result of the base criterion or None, if this is the base criterion.
        :param observation_window: The observation window.
        :return: A DataFrame with the result of the query.
        """
        engine = get_engine()
        query = self.criterion.create_query()
        result = engine.query(query, **bind_params)

        result = self.criterion.process_result(result, base_data, observation_window)

        # merge overlapping/adjacent intervals to reduce the number of intervals
        result = process.union_intervals([result])

        return result

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
        :param base_data: The result of the base criterion (required to add full observation window for patients with
                no data).
        :param observation_window: The observation window.
        :return: A DataFrame with the inverted intervals.
        """
        assert self.expr.is_Not, "Dependency is not a Not expression."

        result = process.invert_intervals(
            data[0],
            base_data,
            observation_window=observation_window,
        )

        return result

    def handle_binary_logical_operator(self, data: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Handles a binary logical operator (And or Or) by merging or intersecting the intervals of the

        :param data: The input data.
        :return: A DataFrame with the merged or intersected intervals.
        """

        if len(data) == 1:
            # if there is only one dependency, return the intervals of that dependency, i.e. no merge/intersect
            return data[0]

        if isinstance(self.expr, (logic.And, logic.NonSimplifiableAnd)):
            result = process.intersect_intervals(data)
        elif isinstance(self.expr, logic.Or):
            result = process.union_intervals(data)
        else:
            raise ValueError(f"Unsupported expression type: {self.expr}")

        return result

    def handle_no_data_preserving_operator(
        self,
        data: list[pd.DataFrame],
        base_data: pd.DataFrame,
        observation_window: TimeRange,
    ) -> pd.DataFrame:
        """
        Handles a NoDataPreservingAnd/Or operator.

        These are used to combine POPULATION, INTERVENTION and POPULATION/INTERVENTION results from different
        population/intervention pairs into a single result (i.e. the full recommendation's POPULATION etc.).

        The POSITIVE intervals are intersected (And) or merged (Or), the NO_DATA intervals are intersected and the
        remaining intervals are set to NEGATIVE.

        :param data: The input data.
        :param base_data: The result of the base criterion.
        :param observation_window: The observation window.
        :return: A DataFrame with the merged intervals.
        """
        assert isinstance(
            self.expr, (logic.NoDataPreservingAnd, logic.NoDataPreservingOr)
        ), "Dependency is not a NoDataPreservingAnd / NoDataPreservingOr expression."

        if isinstance(self.expr, logic.NoDataPreservingAnd):
            result = process.intersect_intervals(data)
        elif isinstance(self.expr, logic.NoDataPreservingOr):
            result = process.union_intervals(data)

        # todo: the only difference between this function and handle_binary_logical_operator is the following lines
        #  - can we merge?
        result_negative = process.complementary_intervals(
            result,
            reference_df=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NEGATIVE,
        )

        result = process.concat_dfs([result, result_negative])

        return result

    def handle_left_dependent_toggle(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        base_data: pd.DataFrame,
        observation_window: TimeRange,
    ) -> pd.DataFrame:
        """
        Handles a left dependent toggle by merging the intervals of the left dependency with the intervals of the
        right dependency according to the following rules:

        - If P is NEGATIVE or NO_DATA, the result is NOT_APPLICABLE (NO_DATA: because we cannot decide whether the
            recommendationis applicable or not).
        - If P is POSITIVE, the result is:
            - POSITIVE if I is POSITIVE
            - NEGATIVE if I is NEGATIVE
            - NO_DATA if I is NO_DATA

        In tabular form:

        | P | I | Result |
        |---|---|--------|
        | NEGATIVE | * | NOT_APPLICABLE |
        | NO_DATA | * | NOT_APPLICABLE |
        | POSITIVE | POSITIVE | POSITIVE |
        | POSITIVE | NEGATIVE | NEGATIVE |
        | POSITIVE | NO_DATA | NO_DATA |

        :param left: The intervals of the left dependency (the one that determines whether the right dependency is
                    returned).
        :param right: The intervals of the right dependency (the one that is taken when the left dependency is
                      POSITIVE).
        :param base_data: The result of the base criterion.
        :param observation_window: The observation window.
        :return: A DataFrame with the merged intervals.
        """
        assert isinstance(
            self.expr, logic.LeftDependentToggle
        ), "Dependency is not a LeftDependentToggle expression."

        # data[0] is the left dependency (i.e. P)
        # data[1] is the right dependency (i.e. I)

        idx_p = left["interval_type"] == IntervalType.POSITIVE
        data_p = left[idx_p]

        result_not_p = process.complementary_intervals(
            data_p,
            reference_df=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NOT_APPLICABLE,
        )

        # P and I --> POSITIVE
        result_p_and_i = process.intersect_intervals([data_p, right])

        result = process.concat_dfs([result_not_p, result_p_and_i])

        # fill remaining time with NEGATIVE
        result_no_data = process.complementary_intervals(
            result,
            reference_df=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NEGATIVE,
        )

        result = process.concat_dfs([result, result_no_data])

        return result

    def store_result_in_db(
        self,
        result: pd.DataFrame,
        base_data: pd.DataFrame | None,
        bind_params: dict,
        observation_window: TimeRange,
    ) -> None:
        """
        Stores the result in the database.

        :param result: The result to store.
        :param base_data: The result of the base criterion.
        :param bind_params: The parameters.
        :param observation_window: The observation window.
        :return: None.
        """
        # todo do we want to assign here (i.e. instead of outside the task somehow or at least as params to the statement)

        if len(result) == 0:
            return

        if base_data is not None:
            # intersect with the base criterion
            result = process.mask_intervals(
                result,
                mask=base_data,
                # interval_type_outside_mask=IntervalType.NOT_APPLICABLE,
                # observation_window=observation_window,
            )

        pi_pair_id = bind_params.get("pi_pair_id", None)
        criterion_id = self.criterion.id if self.expr.is_Atom else None

        if self.expr.is_Atom:
            assert pi_pair_id is None, "pi_pair_id shall be None for criterion"

        result = result.assign(
            criterion_id=criterion_id,
            pi_pair_id=pi_pair_id,
            recommendation_run_id=bind_params["run_id"],
            cohort_category=self.category,
        )
        try:
            with get_engine().begin() as conn:
                conn.execute(
                    RecommendationResultInterval.__table__.insert(),
                    result.to_dict(orient="records"),
                )
        except IntegrityError as e:
            # Handle integrity errors (e.g., constraint violations)
            logging.error(
                f"A database integrity error occurred in task {self.name()}: {e}"
            )
            raise
        except DBAPIError as e:
            # Handle exceptions specific to the DBAPI in use
            logging.error(f"A DBAPI error occurred in task {self.name()}: {e}")
            raise
        except SQLAlchemyError as e:
            # Handle general SQLAlchemy errors
            logging.error(f"A SQLAlchemy error occurred in task {self.name()}: {e}")
            raise
        except Exception as e:
            # Handle other exceptions
            logging.error(f"An error occurred in task {self.name()}: {e}")
            raise

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
            return f"Task({self.expr.name}, criterion={self.criterion}, category={self.expr.category})"
        else:
            return f"Task({self.expr}), category={self.expr.category})"
