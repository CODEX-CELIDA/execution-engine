import base64
import datetime
import json
import logging
from enum import Enum, auto
from typing import List

from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError, SQLAlchemyError

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.temporal import TimeIntervalType
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.omop.sqlclient import OMOPSQLClient
from execution_engine.settings import get_config
from execution_engine.task.process import (
    Interval,
    IntervalWithCount,
    AnyInterval,
    GeneralizedInterval,
    get_processing_module,
    interval_like,
)
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
        expr: logic.Expr,
        criterion: Criterion | None,
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

                assert (
                    self.criterion is not None
                ), "criterion shall not be None for atomic expression"

                logging.debug(f"Running criterion - '{self.name()}'")
                result = self.handle_criterion(
                    self.criterion, bind_params, base_data, observation_window
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
                elif isinstance(
                    self.expr, (logic.LeftDependentToggle, logic.ConditionalFilter)
                ):
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
                elif isinstance(self.expr, logic.TemporalCount):
                    result = self.handle_temporal_operator(data, observation_window)
                else:
                    raise ValueError(f"Unsupported expression type: {type(self.expr)}")

                if self.store_result:
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
            def interval_counts(start: int, end: int, intervals: List[AnyInterval]) -> GeneralizedInterval:
                positive_count = 0
                not_applicable_count = 0
                for interval in intervals:
                    if interval is None or interval.type == IntervalType.NEGATIVE:
                        pass
                    elif interval.type == IntervalType.POSITIVE:
                        positive_count += 1
                    elif interval.type == IntervalType.NOT_APPLICABLE:
                        not_applicable_count += 1
                # we require at least one positive interval to be present in any case (hence the max(1, ...))
                effective_count_min = max(1, self.expr.count_min - not_applicable_count)
                if positive_count >= effective_count_min:
                    effective_type = IntervalType.POSITIVE
                else:
                    effective_type = IntervalType.NEGATIVE
                ratio = positive_count / effective_count_min
                return IntervalWithCount(start, end, effective_type, ratio)
            return process.find_rectangles(data, interval_counts)

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
            def intersection_interval(start: int, end: int, intervals: List[GeneralizedInterval]) -> GeneralizedInterval:
                with IntervalType.intersection_order():
                    min_interval = min(intervals, key = lambda i: IntervalType.NEGATIVE if i is None else i.type)
                if min_interval is not None:
                    return interval_like(min_interval, start, end)
                else:
                    # Explicit representation of negative intervals is
                    # required here because the database views do not
                    # understand the implicit representation.
                    return Interval(start, end, IntervalType.NEGATIVE)
            result = process.find_rectangles(data, intersection_interval)
        else:
            assert isinstance(self.expr, logic.NoDataPreservingOr)
            result = process.union_intervals(data)

        # todo: the only difference between this function and handle_binary_logical_operator is the following lines
        #  - can we merge?
        result_negative = process.complementary_intervals(
            result,
            reference=base_data,
            observation_window=observation_window,
            interval_type=IntervalType.NEGATIVE,
        )

        def union_interval(start: int, end: int, intervals: List[GeneralizedInterval]) -> GeneralizedInterval:
            with IntervalType.union_order():
                max_interval = max(intervals, key=lambda i: IntervalType.NEGATIVE if i is None else i.type)
            if max_interval is not None:
                return interval_like(max_interval, start, end)
            else:
                # Explicit representation of negative intervals is
                # required here because the database views do not
                # understand the implicit representation.
                return Interval(start, end, IntervalType.NEGATIVE)
        result = process.find_rectangles([result, result_negative], union_interval)

        return result

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
        # observation_window_intervals extends the result to the
        # correct temporal range; Its type is not important.
        observation_window_intervals = {key: [Interval(observation_window.start.timestamp(),
                                                       observation_window.end.timestamp(),
                                                       IntervalType.POSITIVE)]
                                        for key in left.keys()}
        if isinstance(self.expr, logic.LeftDependentToggle):
            fill_type = IntervalType.NOT_APPLICABLE
        else:
            assert isinstance(self.expr, logic.ConditionalFilter)
            fill_type = IntervalType.NEGATIVE

        def new_interval(start: int, end: int, intervals: List[GeneralizedInterval]) -> GeneralizedInterval:
            left_interval, right_interval, observation_window_ = intervals
            if (left_interval is None) or not left_interval.type == IntervalType.POSITIVE :
                # no left_interval or not positive -> use fill type
                return Interval(start, end, fill_type)
            elif right_interval is not None:
                return interval_like(right_interval, start, end)
            else: # left_interval but not right_interval -> implicit negative
                return None
        return process.find_rectangles([left, right, observation_window_intervals], new_interval)

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
        assert self.expr.count_min == 1
        assert self.expr.count_max is None

        if self.expr.interval_criterion is not None:
            # last element is the indicator windows
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

            # Create a "temporary window interval" for each window
            # interval. Associate with each temporary window interval
            # all data intervals that overlap it. The association
            # works by assigning a unique id to each temporary window
            # interval.
            ids = dict() # window_interval -> unique id
            infos = dict() # unique id -> list of overlapping data intervals
            def temporary_window_interval(start: int, end: int, intervals: List[AnyInterval]):
                window_interval, data_interval = intervals
                if window_interval is None or window_interval.type == IntervalType.NOT_APPLICABLE:
                    return Interval(start, end, IntervalType.NOT_APPLICABLE)
                else:
                    window_id = ids.get(window_interval, len(ids))
                    ids[window_interval] = window_id
                    info = infos.get(window_id, set())
                    infos[window_id] = info
                    data_interval_type = data_interval.type if data_interval is not None else IntervalType.NEGATIVE
                    info.add(data_interval_type)
                    return IntervalWithCount(start, end, IntervalType.POSITIVE, window_id)
            person_indicator_windows = { key: indicator_windows for key in data_p.keys() }
            result = process.find_rectangles([ person_indicator_windows, data_p], temporary_window_interval)
            # Turn the temporary window intervals into the final
            # intervals by computing the interval types based on the
            # respective overlapping data intervals.
            def finalize_interval(interval):
                if isinstance(interval, IntervalWithCount):
                    window_id = interval.count
                    data_intervals = infos[window_id]
                    # TODO(jmoringe): there should be a way to implement this with max(data_intervals)
                    if IntervalType.POSITIVE in data_intervals:
                        interval_type = IntervalType.POSITIVE
                    elif IntervalType.NEGATIVE in data_intervals:
                        interval_type = IntervalType.NEGATIVE
                    elif IntervalType.NOT_APPLICABLE in data_intervals:
                        interval_type = IntervalType.NOT_APPLICABLE
                    else:
                        assert IntervalType.NO_DATA in data_intervals
                        interval_type = IntervalType.NO_DATA
                    return Interval(interval.lower, interval.upper, interval_type)
                else:
                    return interval
            result = { key: [ finalize_interval(i) for i in intervals ]
                       for key, intervals in result.items() }

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
        criterion_id = self.criterion.id if self.expr.is_Atom else None  # type: ignore # when expr.is_Atom, criterion is not None

        if self.expr.is_Atom:
            assert pi_pair_id is None, "pi_pair_id shall be None for criterion"

        params = dict(
            criterion_id=criterion_id,
            pi_pair_id=pi_pair_id,
            run_id=bind_params["run_id"],
            cohort_category=self.category,
        )
        def interval_data(interval):
            data = dict(
                interval_start=interval.lower,
                interval_end=interval.upper,
                interval_type=interval.type,
            )
            if isinstance(interval, Interval):
                data["interval_ratio"] = None
            else:
                assert isinstance(interval, IntervalWithCount)
                data["interval_ratio"] = interval.count
            return data

        try:
            with get_engine().begin() as conn:
                conn.execute(
                    ResultInterval.__table__.insert(),
                    [
                        {
                            "person_id": person_id,
                            **interval_data(normalized_interval),
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
        return f"[{self.id()}] {str(self)}"

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
        if self.expr.is_Atom:
            return f"Task(criterion={self.expr}, category={self.expr.category})"
        else:
            return f"Task({self.expr}), category={self.expr.category})"
