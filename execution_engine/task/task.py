import base64
import datetime
import json
import logging
from enum import Enum, auto

from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError, SQLAlchemyError

import execution_engine.util.logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.omop.sqlclient import OMOPSQLClient
from execution_engine.settings import get_config
from execution_engine.task.process import Interval, get_processing_module
from execution_engine.util.enum import TimeIntervalType
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import PersonIntervals, TimeRange

process = get_processing_module()


def get_engine() -> OMOPSQLClient:
    """
    Returns a OMOPSQLClient object.
    """
    return OMOPSQLClient(
        **get_config().omop.model_dump(by_alias=True),
        timezone=get_config().timezone,
        null_pool=True,
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

    def __init__(
        self,
        expr: logic.BaseExpr,
        bind_params: dict | None,
        store_result: bool = False,
    ) -> None:
        self.expr = expr
        self.dependencies: list[Task] = []
        self.status = TaskStatus.PENDING
        self.bind_params = bind_params if bind_params is not None else {}
        self.store_result = store_result

    @property
    def category(self) -> CohortCategory:
        """
        Returns the category of the task.
        """
        return self.bind_params["category"]

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

    def select_predecessor_result(
        self, expr: logic.BaseExpr, data: list[PersonIntervals]
    ) -> PersonIntervals:
        """
        Select the result results of the predecessor task from the given expression.

        This is required in expressions where order is important, e.g. in BinaryNonCommutativeOperator.
        As the nx.DiGraph (and by inheritance, ExecutionGraph) does not store the order of the predecessors,
        we need to find the predecessor task by its expression and select the result from the data.

        :param expr: The expression of the predecessor task.
        :param data: The input data.
        :return: The result of the predecessor task.
        """
        if len(self.dependencies) == 0:
            raise ValueError("Task has no dependencies.")

        idx = next((i for i, t in enumerate(self.dependencies) if t.expr == expr), None)

        if idx is None:
            raise ValueError(
                f"Task with expression '{str(expr)}' not found in dependencies."
            )

        return data[idx]

    def run(
        self,
        data: list[PersonIntervals],
        base_data: PersonIntervals | None,
        bind_params: dict,
    ) -> PersonIntervals:
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
        logging.debug(f"Running task '{self.name()}'")

        try:
            if len(self.dependencies) == 0 or self.expr.is_Atom:
                # atomic expressions (i.e. criterion)

                logging.debug(f"Running criterion - '{self.name()}'")

                assert isinstance(self.expr, Criterion), "Dependency is not a Criterion"

                result = self.handle_criterion(
                    self.expr, bind_params, base_data, observation_window
                )

                logging.debug(f"Storing results - '{self.name()}'")
                self.store_result_in_db(result, base_data, bind_params)

            else:
                assert (
                    base_data is not None
                ), "base_data shall not be None for non-atomic expression"

                logging.debug(f"Running dependencies - '{self.name()}'")

                # non-atomic expressions (i.e. logical operations on criteria)
                if isinstance(self.expr, logic.Not):
                    result = self.handle_unary_logical_operator(
                        data, base_data, observation_window
                    )
                elif isinstance(
                    self.expr,
                    (
                        logic.And,
                        logic.Or,
                        logic.NonSimplifiableAnd,
                        logic.Count,
                        logic.CappedCount,
                        logic.AllOrNone,
                    ),
                ):
                    result = self.handle_binary_logical_operator(data)
                elif isinstance(self.expr, (logic.BinaryNonCommutativeOperator)):
                    result = self.handle_left_dependent_toggle(
                        left=self.select_predecessor_result(self.expr.left, data),
                        right=self.select_predecessor_result(self.expr.right, data),
                        base_data=base_data,
                        observation_window=observation_window,
                    )
                elif isinstance(
                    self.expr, (logic.NoDataPreservingAnd, logic.NoDataPreservingOr)
                ):
                    result = self.handle_no_data_preserving_operator(
                        data, base_data, observation_window
                    )
                elif isinstance(self.expr, logic.TemporalCount):
                    result = self.handle_temporal_operator(data, observation_window)
                else:
                    raise ValueError(f"Unsupported expression type: {type(self.expr)}")

                if self.store_result:
                    if (
                        not isinstance(self.expr, logic.NoDataPreservingAnd)
                        and not self.expr.is_Atom
                    ):
                        result = self.insert_negative_intervals(
                            result, base_data, observation_window
                        )
                    logging.debug(f"Storing results - '{self.name()}'")
                    self.store_result_in_db(result, base_data, bind_params)

        except Exception as e:
            self.status = TaskStatus.FAILED
            exception_type = type(e).__name__  # Get the type of the exception
            raise TaskError(
                f"Task '{self.name()}' failed with error: {exception_type}"
            ) from e

        self.status = TaskStatus.COMPLETED

        return result

    @classmethod
    def handle_criterion(
        cls,
        criterion: Criterion,
        bind_params: dict,
        base_data: PersonIntervals | None,
        observation_window: TimeRange,
    ) -> PersonIntervals:
        """
        Handles a criterion by querying the database.

        :param criterion: The criterion to handle.
        :param bind_params: The parameters.
        :param base_data: The result of the base criterion or None, if this is the base criterion.
        :param observation_window: The observation window.
        :return: A DataFrame with the result of the query.
        """
        engine = get_engine()
        query = criterion.create_query()
        engine.log_query(query, params=bind_params)

        logging.debug(f"Running query - '{criterion.description()}'")
        result = engine.raw_query(query, params=bind_params)

        # merge overlapping/adjacent intervals to reduce the number of intervals - but NEGATIVE is dominant over
        # POSITIVE here, i.e. if there is a NEGATIVE interval, the result is NEGATIVE, regardless of any POSITIVE
        # intervals at the same time
        logging.debug(f"Merging intervals - '{criterion.description()}'")
        with IntervalType.custom_union_priority_order(
            IntervalType.intersection_priority()
        ):
            data = process.result_to_intervals(result)

        logging.debug(f"Processing data - '{criterion.description()}'")
        data = criterion.process_data(data, base_data, observation_window)

        return data

    def handle_unary_logical_operator(
        self,
        data: list[PersonIntervals],
        base_data: PersonIntervals,
        observation_window: TimeRange,
    ) -> PersonIntervals:
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

    def handle_binary_logical_operator(
        self, data: list[PersonIntervals]
    ) -> PersonIntervals:
        """
        Handles a binary logical operator (And or Or) by merging or intersecting the intervals of the

        :param data: The input data.
        :return: A DataFrame with the merged or intersected intervals.
        """

        if (
            isinstance(self.expr, (logic.And, logic.NonSimplifiableAnd, logic.Or))
            and len(data) == 1
        ):
            # if there is only one dependency, return the intervals of that dependency, i.e. no merge/intersect
            return data[0]

        if isinstance(self.expr, (logic.And, logic.NonSimplifiableAnd)):
            result = process.intersect_intervals(data)
        elif isinstance(self.expr, logic.Or):
            result = process.union_intervals(data)
        elif isinstance(self.expr, logic.Count):
            result = process.count_intervals(data)
            result = process.filter_count_intervals(
                result,
                min_count=self.expr.count_min,
                max_count=self.expr.count_max,
            )
        elif isinstance(self.expr, logic.CappedCount):
            intervals_with_count = process.find_rectangles_with_count(data)
            result = dict()
            for key, intervals in intervals_with_count.items():

                key_result = []

                for interval in intervals:
                    counts = interval.counts
                    not_applicable_count = counts.get(IntervalType.NOT_APPLICABLE, 0)

                    # we require at least one positive interval to be present in any case (hence the max(1, ...))
                    effective_count_min = max(
                        1, self.expr.count_min - not_applicable_count
                    )
                    positive_count = counts.get(IntervalType.POSITIVE, 0)
                    effective_type = (
                        IntervalType.POSITIVE
                        if positive_count >= effective_count_min
                        else IntervalType.NEGATIVE
                    )
                    key_result.append(
                        Interval(interval.lower, interval.upper, effective_type)
                    )
                result[key] = key_result
            return result
        elif isinstance(self.expr, logic.AllOrNone):
            raise NotImplementedError("AllOrNone is not implemented yet.")
        else:
            raise ValueError(f"Unsupported expression type: {self.expr}")

        return result

    def handle_no_data_preserving_operator(
        self,
        data: list[PersonIntervals],
        base_data: PersonIntervals,
        observation_window: TimeRange,
    ) -> PersonIntervals:
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
        else:
            raise ValueError(f"Unsupported expression type: {type(self.expr)}")

        # todo: the only difference between this function and handle_binary_logical_operator is the following lines
        #  - can we merge?
        return self.insert_negative_intervals(
            data=result, base_data=base_data, observation_window=observation_window
        )

    def handle_left_dependent_toggle(
        self,
        left: PersonIntervals,
        right: PersonIntervals,
        base_data: PersonIntervals,
        observation_window: TimeRange,
    ) -> PersonIntervals:
        """
        Handles a left dependent toggle or a conditional filter by merging the intervals of the left dependency with the
        intervals of the right dependency according to the following rules. The difference between the two is that
        the conditional filter yields NEGATIVE if left dependence is NEGATIVE or NO_DATA, while the left dependent
        toggle yields NOT_APPLICABLE in these cases.

        LeftDependentToggle
        -------------------
        - If P is NEGATIVE or NO_DATA, the result is NOT_APPLICABLE (NO_DATA: because we cannot decide whether the
            recommendation is applicable or not).
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

        ConditionalFilter
        -------------------
        - If P is NEGATIVE or NO_DATA, the result is NEGATIVE.
        - If P is POSITIVE, the result is:
            - POSITIVE if I is POSITIVE
            - NEGATIVE if I is NEGATIVE
            - NO_DATA if I is NO_DATA

        In tabular form:

        | P | I | Result |
        |---|---|--------|
        | NEGATIVE | * | NEGATIVE |
        | NO_DATA | * | NEGATIVE |
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
            self.expr, (logic.LeftDependentToggle, logic.ConditionalFilter)
        ), "Dependency is not a LeftDependentToggle or ConditionalFilter expression."

        # data[0] is the left dependency (i.e. P)
        # data[1] is the right dependency (i.e. I)

        data_p = process.select_type(left, IntervalType.POSITIVE)

        if isinstance(self.expr, logic.LeftDependentToggle):
            interval_type = IntervalType.NOT_APPLICABLE
        elif isinstance(self.expr, logic.ConditionalFilter):
            interval_type = IntervalType.NEGATIVE

        result_not_p = process.complementary_intervals(
            data_p,
            reference=base_data,
            observation_window=observation_window,
            interval_type=interval_type,
        )

        result_p_and_i = process.intersect_intervals([data_p, right])

        result = process.concat_intervals([result_not_p, result_p_and_i])

        # fill remaining time with NEGATIVE
        result_no_data = process.complementary_intervals(
            result,
            reference=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NEGATIVE,
        )

        result = process.concat_intervals([result, result_no_data])

        return result

    def handle_temporal_operator(
        self, data: list[PersonIntervals], observation_window: TimeRange
    ) -> PersonIntervals:
        """
        Handles a TemporalCount operator.

        May be used to aggregate multiple criteria in a temporal manner, e.g. to count the number of times a certain
        condition is met within a certain time frame (e.g. morning shift).

        :param data: The input data.
        :param observation_window: The observation window.
        :return: A DataFrame with the merged intervals.
        """

        data_p = data[0]
        # data_p = process.select_type(data[0], IntervalType.POSITIVE)
        # data_p = {key: val for key, val in data_p.items() if val}

        def get_start_end_from_interval_type(
            type_: TimeIntervalType,
        ) -> tuple[datetime.time, datetime.time]:
            """
            Returns the start and end time for a given TimeIntervalType, read from the configuration.
            """
            try:
                cnf = getattr(get_config().time_intervals, type_.value)
            except AttributeError:
                raise ValueError(f"No time interval settings for {type_}")
            return cnf.start, cnf.end

        assert isinstance(self.expr, logic.TemporalCount), "Invalid expression type"

        if self.expr.interval_criterion is not None:

            # last element is the indicator windows
            assert (
                len(data) >= 2
            ), "TemporalCount with indicator criterion requires at least two inputs"
            data, indicator_personal_windows = data[:-1], data[-1]

            result = process.find_overlapping_personal_windows(
                indicator_personal_windows, data_p
            )
        else:

            if self.expr.interval_type == TimeIntervalType.ANY_TIME:
                indicator_windows = [
                    Interval(
                        lower=observation_window.start.timestamp(),
                        upper=observation_window.end.timestamp(),
                        type=IntervalType.POSITIVE,
                    )
                ]
            else:
                if self.expr.interval_type is not None:
                    start_time, end_time = get_start_end_from_interval_type(
                        self.expr.interval_type
                    )
                elif (
                    self.expr.start_time is not None and self.expr.end_time is not None
                ):
                    start_time, end_time = self.expr.start_time, self.expr.end_time
                else:
                    raise ValueError("Invalid time interval settings")

                indicator_windows = process.create_time_intervals(
                    start_datetime=observation_window.start,
                    end_datetime=observation_window.end,
                    start_time=start_time,
                    end_time=end_time,
                    interval_type=IntervalType.POSITIVE,
                    timezone=get_config().timezone,
                )

            result = process.find_overlapping_windows(indicator_windows, data_p)

        return result

    def insert_negative_intervals(
        self,
        data: PersonIntervals,
        base_data: PersonIntervals,
        observation_window: TimeRange,
    ) -> PersonIntervals:
        """
        Inserts negative intervals into the result.

        Usually, negative intervals are implicit. This functions fills all gaps between other intervals with negative
        intervals.

        :param data: The input data.
        :param base_data: The result of the base criterion.
        :param observation_window: The observation window.
        :return: A DataFrame with the merged intervals.
        """

        data_negative = process.complementary_intervals(
            data,
            reference=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NEGATIVE,
        )

        result = process.concat_intervals([data, data_negative])

        return result

    def store_result_in_db(
        self,
        result: PersonIntervals,
        base_data: PersonIntervals | None,
        bind_params: dict,
    ) -> None:
        """
        Stores the result in the database.

        :param result: The result to store.
        :param base_data: The result of the base criterion.
        :param bind_params: The parameters.
        :return: None.
        """
        if not result:
            return

        if base_data is not None:
            # intersect with the base criterion
            result = process.mask_intervals(
                result,
                mask=base_data,
            )

            if not result:
                return

        pi_pair_id = bind_params.get("pi_pair_id", None)
        criterion_id = self.expr.id if self.expr.is_Atom else None  # type: ignore # when expr.is_Atom, criterion is not None

        if self.expr.is_Atom:
            assert pi_pair_id is None, "pi_pair_id shall be None for criterion"

        params = dict(
            criterion_id=criterion_id,
            pi_pair_id=pi_pair_id,
            run_id=bind_params["run_id"],
            cohort_category=self.category,
        )

        try:
            with get_engine().begin() as conn:
                conn.execute(
                    ResultInterval.__table__.insert(),
                    [
                        {
                            "person_id": person_id,
                            "interval_start": normalized_interval.lower,
                            "interval_end": normalized_interval.upper,
                            "interval_type": normalized_interval.type,
                            **params,
                        }
                        for person_id, intervals in result.items()
                        for interval in intervals
                        for normalized_interval in [
                            process.normalize_interval(interval)
                        ]
                    ],
                )
        except ProgrammingError as e:
            # Handle programming errors (e.g., syntax errors)
            logging.error(f"A programming error occurred in task {self.name()}: {e}")
            raise
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
        Returns the (unique) name of the Task object.

        Uniqueness is guaranteed by prepending the base64-encoded hash of the Task object.
        """
        if self.expr.is_Atom:
            return f"[{self.id()}] {str(self)}"
        else:
            return f"[{self.id()}] {self.expr.__class__.__name__}()"

    def id(self) -> str:
        """
        Returns the id of the Task object.
        """
        hash_value = hash((str(self.expr), json.dumps(self.bind_params)))

        # Determine the number of bytes needed. Python's hash returns a value based on the platform's pointer size.
        # It's 8 bytes for 64-bit systems and 4 bytes for 32-bit systems.
        num_bytes = 8 if hash_value.bit_length() > 32 else 4

        # Convert the hash to bytes. Specify 'big' for big-endian and 'signed=True' to handle negative values.
        hash_bytes = hash_value.to_bytes(num_bytes, byteorder="big", signed=True)

        # Base64 encode and return the result
        return base64.b64encode(hash_bytes).decode()

    def __hash__(self) -> int:
        """
        Returns the hash of the Task object.
        """
        return hash((self.expr, json.dumps(self.bind_params)))

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object.
        """
        return f"Task({self.expr}), category={self.category})"
