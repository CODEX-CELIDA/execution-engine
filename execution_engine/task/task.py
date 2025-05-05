import base64
import copy
import datetime
import json
import logging
from collections import Counter
from enum import Enum, auto
from typing import Callable, List, Type, cast

from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError, SQLAlchemyError

import execution_engine.util.logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.omop.sqlclient import OMOPSQLClient, datetime_cols_to_epoch
from execution_engine.settings import get_config
from execution_engine.task.process import (
    AnyInterval,
    GeneralizedInterval,
    Interval,
    IntervalWithCount,
    get_processing_module,
    interval_like,
    timerange_to_interval,
)
from execution_engine.util.enum import TimeIntervalType
from execution_engine.util.interval import IntervalType
from execution_engine.util.types import PersonIntervals
from execution_engine.util.types.timerange import TimeRange

process = get_processing_module()

COUNT_TYPES = (
    logic.MinCount,
    logic.ExactCount,
    logic.CappedMinCount,
)


def default_interval_union_with_count(
    start: int, end: int, intervals: List[GeneralizedInterval]
) -> IntervalWithCount:
    """
    Default interval counting function to be used in logic.Or
    """
    result_type = None
    result_count = 0
    for interval in intervals:
        if interval is None:
            interval_type, interval_count = IntervalType.NEGATIVE, 0
        else:
            interval_type, interval_count = interval.type, cast(int, interval.count)

        if (
            (
                interval_type is IntervalType.POSITIVE
                and result_type is not IntervalType.POSITIVE
            )
            or (
                interval_type is IntervalType.NO_DATA
                and result_type is not IntervalType.POSITIVE
                and result_type is not IntervalType.NO_DATA
            )
            or (
                interval_type is IntervalType.NEGATIVE
                and (result_type is IntervalType.NOT_APPLICABLE or result_type is None)
            )
            or (interval_type is IntervalType.NOT_APPLICABLE and result_type is None)
        ):
            result_type = interval_type
            result_count = 0
        result_count += interval_count
    return IntervalWithCount(start, end, result_type, result_count)


def default_interval_intersect_with_count(
    start: int, end: int, intervals: List[GeneralizedInterval]
) -> IntervalWithCount:
    """
    Default interval counting function to be used in logic.Or
    """
    raise NotImplementedError()


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

    def get_predecessor_data_index(self, expr: logic.BaseExpr) -> int:
        """
        Get the index of the predecessor data from the given expression.

        This is required in expressions where order is important, e.g. in BinaryNonCommutativeOperator.
        As the nx.DiGraph (and by inheritance, ExecutionGraph) does not store the order of the predecessors,
        we need to find the predecessor task by its expression and select the result from the data.

        :param expr: The expression of the predecessor task.
        :return: The index in of expr in the data of predecessor results
        """
        if len(self.dependencies) == 0:
            raise ValueError("Task has no dependencies.")

        idx = next((i for i, t in enumerate(self.dependencies) if t.expr == expr), None)

        if idx is None:
            raise ValueError(
                f"Task with expression '{str(expr)}' not found in dependencies."
            )

        return idx

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
        return data[self.get_predecessor_data_index(expr)]

    def receives_only_count_inputs(self) -> bool:
        """
        Indicates whether this tasks only receives inputs from expression that perform counting and thus return
        IntervalWithCount.
        """
        #  all arguments are either count types or logic.BinaryNonCommutativeOperator
        #  with their "right" child being a count type, or they have a custom counting function (count_intervals())
        if all(
            (
                isinstance(parent, logic.BinaryNonCommutativeOperator)
                and isinstance(parent.right, COUNT_TYPES)
            )
            or (isinstance(parent, COUNT_TYPES))
            or (hasattr(parent, "count_intervals"))
            for parent in self.expr.args
        ):
            return True

        return False

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
                elif isinstance(self.expr, logic.TemporalCount):
                    result = self.handle_temporal_operator(data, observation_window)
                elif isinstance(
                    self.expr,
                    (logic.CommutativeOperator),
                ):
                    result = self.handle_binary_logical_operator(data)
                elif isinstance(self.expr, logic.BinaryNonCommutativeOperator):
                    result = self.handle_left_dependent_toggle(
                        left=self.select_predecessor_result(self.expr.left, data),
                        right=self.select_predecessor_result(self.expr.right, data),
                        base_data=base_data,
                        observation_window=observation_window,
                    )
                else:
                    raise ValueError(f"Unsupported expression type: {type(self.expr)}")

                if self.store_result:
                    if not self.expr.is_Atom:
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
        query = datetime_cols_to_epoch(query)
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
        assert len(data) == 1, "Unary operators require only one input"

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
        Handles a binary logical operator by using the appropriate processing function.

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

            if self.receives_only_count_inputs() and hasattr(
                self.expr, "count_intervals"
            ):
                # we check if there are custom data preparatory and interval counting functions and use these
                prepare_func = getattr(self.expr, "prepare_data", None)
                if prepare_func:
                    data = prepare_func(self, data)
                func = getattr(
                    self.expr, "count_intervals", default_interval_intersect_with_count
                )
                result = process.find_rectangles(data, func)

            else:
                result = process.intersect_intervals(data)

        elif isinstance(self.expr, (logic.Or, logic.NonSimplifiableOr)):
            if self.receives_only_count_inputs() and hasattr(
                self.expr, "count_intervals"
            ):
                # we check if there are custom data preparatory and interval counting functions and use these
                prepare_func = getattr(self.expr, "prepare_data", None)
                if prepare_func:
                    data = prepare_func(self, data)
                func = getattr(
                    self.expr, "count_intervals", default_interval_union_with_count
                )
                result = process.find_rectangles(data, func)
            else:
                result = process.union_intervals(data)

        elif isinstance(self.expr, logic.Count):
            count_min = self.expr.count_min
            count_max = self.expr.count_max
            if count_min is None and count_max is None:
                raise ValueError("count_min and count_max cannot both be None")
            if count_min is None:
                count_min = 0

            def interval_counts(
                start: int, end: int, intervals: List[GeneralizedInterval]
            ) -> GeneralizedInterval:

                # Count the different interval types. None represents
                # implicit negative intervals and is counted as such.
                counts = Counter(
                    (interval.type if interval else IntervalType.NEGATIVE)
                    for interval in intervals
                )

                # Either the count constraints or the interval type
                # with the highest "union priority" determines the
                # result.
                positive_count = counts[IntervalType.POSITIVE]
                if positive_count > 0 or count_min == 0:
                    if count_min == 0:
                        if positive_count <= count_max:  # type: ignore[operator]
                            return Interval(start, end, IntervalType.POSITIVE)
                        else:
                            return None  # Implicit negative interval
                    else:
                        min_good = count_min <= positive_count
                        max_good = (count_max is None) or (positive_count <= count_max)
                        interval_type = (
                            IntervalType.POSITIVE
                            if (min_good and max_good)
                            else IntervalType.NEGATIVE
                        )
                        ratio = positive_count / count_min
                        return IntervalWithCount(start, end, interval_type, ratio)
                if counts[IntervalType.NO_DATA] > 0:
                    return IntervalWithCount(start, end, IntervalType.NO_DATA, 0)
                if counts[IntervalType.NOT_APPLICABLE] > 0:
                    return IntervalWithCount(start, end, IntervalType.NOT_APPLICABLE, 0)
                if counts[IntervalType.NEGATIVE] > 0:
                    return IntervalWithCount(start, end, IntervalType.NEGATIVE, 0)

                raise ValueError("No intervals of any kind found")

            result = process.find_rectangles(data, interval_counts)

        elif isinstance(self.expr, logic.CappedCount):

            def interval_counts(
                start: int, end: int, intervals: List[AnyInterval]
            ) -> GeneralizedInterval:
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
                effective_count_min = min(
                    self.expr.count_min, max(1, len(intervals) - not_applicable_count)  # type: ignore[attr-defined]
                )
                if positive_count >= effective_count_min:
                    effective_type = IntervalType.POSITIVE
                else:
                    effective_type = IntervalType.NEGATIVE
                ratio = positive_count / effective_count_min
                return IntervalWithCount(start, end, effective_type, ratio)

            result = process.find_rectangles(data, interval_counts)

        elif isinstance(self.expr, logic.AllOrNone):
            raise NotImplementedError("AllOrNone is not implemented yet.")
        else:
            raise ValueError(f"Unsupported expression type: {self.expr}")

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

        # window_intervals extends the result to the correct temporal
        # range; Its type is not important.
        # use a tuple for windows to make sure it is immutable (and can be shared by all persons)
        windows = (
            timerange_to_interval(observation_window, type_=IntervalType.POSITIVE),
        )
        window_intervals = {key: windows for key in left.keys()}

        if isinstance(self.expr, logic.LeftDependentToggle):
            fill_type = IntervalType.NOT_APPLICABLE
        else:
            assert isinstance(self.expr, logic.ConditionalFilter)
            fill_type = IntervalType.NEGATIVE

        interval_type: (
            Callable[[int, int, IntervalType], IntervalWithCount] | Type[Interval]
        )

        # if all incoming data of "right" are count types, we will create a count type as well, to
        # allow summing in the next layer
        if isinstance(self.expr.right, logic.Expr) and all(
            isinstance(parent, COUNT_TYPES) for parent in self.expr.right.args
        ):
            interval_type = lambda start, end, fill_type: IntervalWithCount(
                start, end, fill_type, None
            )
        else:
            interval_type = Interval

        def new_interval(
            start: int, end: int, intervals: List[GeneralizedInterval]
        ) -> GeneralizedInterval:
            left_interval, right_interval, observation_window_ = intervals
            if (left_interval is None) or left_interval.type != IntervalType.POSITIVE:
                # no left_interval or not positive -> use fill type
                return interval_type(start, end, fill_type)
            elif right_interval is not None:
                return interval_like(right_interval, start, end)
            else:  # left_interval but not right_interval -> implicit negative
                return None

        return process.find_rectangles([left, right, window_intervals], new_interval)

    def handle_temporal_operator(
        self, data: list[PersonIntervals], observation_window: TimeRange
    ) -> PersonIntervals:
        """
        Handles a TemporalCount operator, which checks whether a certain condition
        (e.g., an event or observation) occurs a specific number of times in each defined test interval.

        Note: Currently, only  TemporalMinCount(*, threshold=1) is supported.

        This method can do one of two main things:
          1) If `self.expr.interval_criterion` is set, it will intersect ("find overlapping")
             a set of personal indicator windows with your data, returning intervals that
             represent when the criterion is met within those windows.
          2) If `interval_criterion` is not set, it will create time slices (intervals) for
             the specified `TimeIntervalType` (e.g., morning shift) within `observation_window`
             and determine whether the data meets the condition inside each slice.

        For instance, if your expression says 'TemporalCount(ANY_TIME, count_min=1)', then
        the entire observation_window is treated as one big interval, and we only need to
        check if we see at least one positive data interval in that timeframe.

        :param data: A list of dictionaries mapping person ID -> list of intervals.
            Typically, the intervals from your workflow or dataset.
        :param observation_window: The overall time range we’re interested in.
        :return: A dictionary mapping each person to a list of resulting intervals
            (e.g., each labeled with POSITIVE, NEGATIVE, NOT_APPLICABLE, etc.).
        """
        assert isinstance(self.expr, logic.TemporalCount)

        if self.expr.interval_criterion is not None:
            # If we have an interval_criterion, we expect exactly two data streams:
            # (1) The "main" data
            # (2) The "indicator" data (e.g., personal windows or ICU periods)
            assert (
                len(data) == 2
            ), f"TemporalCount with indicator criterion requires exactly two input streams, got {len(data)}"

            indicator_personal_windows = data.pop(
                self.get_predecessor_data_index(self.expr.interval_criterion)
            )

        assert (
            len(data) == 1
        ), f"TemporalCount requires exactly one input streams, got {len(data)}"

        data_arg = data[0]

        def get_start_end_from_interval_type(
            type_: TimeIntervalType,
        ) -> tuple[datetime.time, datetime.time]:
            """
            Reads the config for the given TimeIntervalType, returning its start and end times.

            For example, 'MORNING_SHIFT' might map to 06:00 - 14:00, if configured that way.
            """
            try:
                cnf = getattr(get_config().time_intervals, type_.value)
            except AttributeError:
                raise ValueError(f"No time interval settings for {type_}")
            return cnf.start, cnf.end

        assert isinstance(self.expr, logic.TemporalCount), "Invalid expression type"

        if self.expr.count_min != 1 or self.expr.count_max is not None:
            raise NotImplementedError(
                "Currently, only  TemporalMinCount(*, threshold=1) is supported."
            )

        if self.expr.interval_criterion is not None:
            # Filter out only the POSITIVE intervals from the data
            data_positive = process.select_type(data_arg, IntervalType.POSITIVE)

            # Overlap the personal windows with the data intervals to see if the condition
            # is met for each relevant chunk in the personal window.
            result = process.find_overlapping_personal_windows(
                indicator_personal_windows, data_positive
            )
        else:
            # If we have no `interval_criterion`, we must create the intervals ourselves
            # from the observation_window (e.g. ANY_TIME vs MORNING_SHIFT).
            if self.expr.interval_type == TimeIntervalType.ANY_TIME:
                # Just one interval covering the entire observation window
                indicator_windows = (
                    timerange_to_interval(
                        observation_window, type_=IntervalType.POSITIVE
                    ),
                )
            else:
                # If we do have a known interval type, or explicit start/end times, build them:
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

                # Create repeated intervals for each day from observation_window.start
                # up to observation_window.end, using e.g. "06:00 - 14:00" if it's morning, etc.
                indicator_windows = process.create_time_intervals(
                    start_datetime=observation_window.start,
                    end_datetime=observation_window.end,
                    start_time=start_time,
                    end_time=end_time,
                    interval_type=IntervalType.POSITIVE,
                    timezone=get_config().timezone,
                )

            # We'll track each "window interval" by its object ID, storing the best
            # result type found so far (NEGATIVE, POSITIVE, or NOT_APPLICABLE).
            window_types: dict[int, IntervalType] = dict()

            def update_window_type(
                window_interval: GeneralizedInterval, data_interval: GeneralizedInterval
            ) -> IntervalType:
                """
                Called whenever we cross a boundary in time.

                window_interval: The current "indicator" interval (e.g., a morning slice).
                data_interval:   The data interval from data_arg that overlaps with this moment,
                                 could be POSITIVE, NEGATIVE, NO_DATA, NOT_APPLICABLE (or None,
                                 which is interpreted as NEGATIVE).

                This function decides how to update the 'window interval type' based on new info.
                """
                current_type = window_types.get(
                    id(window_interval), IntervalType.NOT_APPLICABLE
                )

                # If the data interval is NEGATIVE (or equivalently, None) or NO_DATA, we treat it as NEGATIVE
                if (
                    data_interval is None
                    or data_interval.type is IntervalType.NO_DATA
                    or data_interval.type is IntervalType.NEGATIVE
                ):
                    # Set current_type to NEGATIVE if it is not already POSITIVE (because a POSITIVE data interval
                    # was passed earlier)
                    if current_type is IntervalType.NOT_APPLICABLE:
                        current_type = IntervalType.NEGATIVE

                elif data_interval.type is IntervalType.POSITIVE:
                    # If the data is POSITIVE, set the window to POSITIVE, overriding any negative state.
                    current_type = IntervalType.POSITIVE

                window_types[id(window_interval)] = current_type

                return current_type

            def is_same_interval(
                left_intervals: List[GeneralizedInterval],
                right_intervals: List[GeneralizedInterval],
            ) -> bool:
                """
                Helper used by ` process.find_rectangles()`.

                The framework calls this to decide if two adjacent intervals share
                the same "payload" and can be merged into a single interval.

                'left_intervals' and 'right_intervals' are each a pair [window_interval, data_interval],
                for the previous block vs. the new block. We update the 'window_types' dict with
                whatever is found in the new block, and return True if we can keep merging them.
                """
                left_window_interval, left_data_interval = left_intervals
                right_window_interval, right_data_interval = right_intervals

                # If the next block's window interval is None, it means we're off-limits
                # or there's no overlap. We update the left block's type and return False.
                if right_window_interval is None:
                    if left_window_interval is None:
                        return True
                    else:
                        update_window_type(left_window_interval, left_data_interval)
                        return False
                else:
                    # We do have a right_window_interval, so update it with the right_data_interval
                    update_window_type(right_window_interval, right_data_interval)

                    # If the left window is None, can't be the same interval
                    if left_window_interval is None:
                        return False
                    else:
                        # If left_window_interval and right_window_interval are literally
                        # the same object, we can treat them as the same interval
                        if left_window_interval is right_window_interval:
                            return True
                        else:
                            # They’re different intervals => finalize left and move on
                            update_window_type(left_window_interval, left_data_interval)
                            return False

            def result_interval(
                start: int, end: int, intervals: List[AnyInterval]
            ) -> AnyInterval:
                """
                Called at the end of building each interval, to produce the final interval
                object that will go into the result.

                'intervals' is [window_interval, data_interval], where either can be None.

                We look up window_interval in window_types to see whether we've determined
                it is POSITIVE, NEGATIVE, or NOT_APPLICABLE.
                """
                window_interval, data_interval = intervals
                if (
                    window_interval is None
                    or window_interval.type is IntervalType.NOT_APPLICABLE
                ):
                    return Interval(start, end, IntervalType.NOT_APPLICABLE)
                else:
                    window_type = window_types.get(id(window_interval), None)
                    if window_type is None:
                        window_type = update_window_type(window_interval, data_interval)
                    return Interval(start, end, window_type)

            # Make separate copies of the intervals for each person so
            # that the object identity of each interval is unique and
            # can be used as a dictionary key.
            person_indicator_windows = {
                key: [copy.copy(window) for window in indicator_windows]
                for key in data_arg.keys()
            }

            result = process.find_rectangles(
                [person_indicator_windows, data_arg],
                result_interval,
                is_same_result=is_same_interval,
            )

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
        # window_intervals extends the result to the correct temporal
        # range and forces results to be computed for patients that
        # are not represented in data; The interval types in
        # window_intervals are not important.
        # use a tuple for windows to make sure it is immutable (and can be shared by all persons)
        windows = (
            timerange_to_interval(observation_window, type_=IntervalType.POSITIVE),
        )
        all_keys = data.keys() | base_data.keys()
        window_intervals = {key: windows for key in all_keys}

        def create_interval(
            start: int, end: int, intervals: List[GeneralizedInterval]
        ) -> GeneralizedInterval:
            interval, window_interval = intervals
            if interval is not None:
                return interval_like(interval, start, end)
            else:
                # Explicit representation of negative intervals is
                # required here because the database views do not
                # understand the implicit representation.
                return Interval(start, end, IntervalType.NEGATIVE)

        return process.find_rectangles([data, window_intervals], create_interval)

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

        def interval_data(interval: AnyInterval) -> dict:
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
        if self.expr.is_Atom:
            return f"[{self.id()}] {str(self)}"
        else:
            return f"[{self.id()}] {self.expr.__class__.__name__}()"

    def id(self) -> str:
        """
        Returns the id of the Task object.
        """
        hash_value = hash((self.expr, json.dumps(self.bind_params)))

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
