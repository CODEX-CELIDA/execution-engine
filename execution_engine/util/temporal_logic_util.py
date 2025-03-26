from datetime import time

from execution_engine.util import logic
from execution_engine.util.enum import TimeIntervalType
from execution_engine.util.interval import IntervalType


def Presence(
    criterion: logic.BaseExpr,
    *,
    interval_type: TimeIntervalType | None = None,
    start_time: time | None = None,
    end_time: time | None = None,
    interval_criterion: logic.BaseExpr | None = None,
) -> logic.TemporalMinCount:
    """
    Create a presence combination of criteria.
    """
    return logic.TemporalMinCount(
        criterion,
        threshold=1,
        interval_type=interval_type,
        start_time=start_time,
        end_time=end_time,
        interval_criterion=interval_criterion,
        result_for_not_applicable=IntervalType.NEGATIVE,
    )


def MinCount(
    criterion: logic.BaseExpr,
    *,
    threshold: int,
    interval_type: TimeIntervalType | None = None,
    start_time: time | None = None,
    end_time: time | None = None,
    interval_criterion: logic.BaseExpr | None = None,
) -> logic.TemporalMinCount:
    """
    Create a minimum count combination of criteria.
    """
    return logic.TemporalMinCount(
        criterion,
        threshold=threshold,
        interval_type=interval_type,
        start_time=start_time,
        end_time=end_time,
        interval_criterion=interval_criterion,
    )


def MaxCount(
    criterion: logic.BaseExpr,
    *,
    threshold: int,
    interval_type: TimeIntervalType | None = None,
    start_time: time | None = None,
    end_time: time | None = None,
    interval_criterion: logic.BaseExpr | None = None,
) -> logic.TemporalMaxCount:
    """
    Create a maximum count combination of criteria.
    """
    return logic.TemporalMaxCount(
        criterion,
        threshold=threshold,
        interval_type=interval_type,
        start_time=start_time,
        end_time=end_time,
        interval_criterion=interval_criterion,
    )


def ExactCount(
    criterion: logic.BaseExpr,
    *,
    threshold: int,
    interval_type: TimeIntervalType | None = None,
    start_time: time | None = None,
    end_time: time | None = None,
    interval_criterion: logic.BaseExpr | None = None,
) -> logic.TemporalExactCount:
    """
    Create an exact count combination of criteria.
    """
    return logic.TemporalExactCount(
        criterion,
        threshold=threshold,
        interval_type=interval_type,
        start_time=start_time,
        end_time=end_time,
        interval_criterion=interval_criterion,
    )


def MorningShift(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create a morning shift combination of criteria.
    """
    return Presence(criterion, interval_type=TimeIntervalType.MORNING_SHIFT)


def AfternoonShift(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create an afternoon shift combination of criteria.
    """
    return Presence(criterion, interval_type=TimeIntervalType.AFTERNOON_SHIFT)


def NightShift(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create a night shift combination of criteria.
    """
    return Presence(criterion, interval_type=TimeIntervalType.NIGHT_SHIFT)


def NightShiftBeforeMidnight(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create a night shift before midnight combination of criteria.
    """
    return Presence(
        criterion, interval_type=TimeIntervalType.NIGHT_SHIFT_BEFORE_MIDNIGHT
    )


def NightShiftAfterMidnight(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create a night shift after midnight combination of criteria.
    """
    return Presence(
        criterion, interval_type=TimeIntervalType.NIGHT_SHIFT_AFTER_MIDNIGHT
    )


def Day(
    criterion: logic.BaseExpr,
) -> logic.TemporalMinCount:
    """
    Create a day combination of criteria.
    """
    return Presence(criterion, interval_type=TimeIntervalType.DAY)


def AnyTime(criterion: logic.BaseExpr) -> logic.TemporalMinCount:
    """
    Any time overlap
    """
    return Presence(criterion, interval_type=TimeIntervalType.ANY_TIME)
