import datetime

from portion import CLOSED, AbstractDiscreteInterval


class IntInterval(AbstractDiscreteInterval):
    """
    An integer interval.
    """

    _step = 1


class DateTimeInterval(AbstractDiscreteInterval):
    """
    A datetime interval (second precision).
    """

    _step = datetime.timedelta(seconds=1)


def interval_int(lower: int, upper: int) -> IntInterval:
    """
    Creates a new integer interval.

    :param lower: The lower bound.
    :param upper: The upper bound.
    :return: The new integer interval.
    """
    return IntInterval.from_atomic(CLOSED, lower, upper, CLOSED)


def empty_interval_int() -> IntInterval:
    """
    Creates an empty integer interval.

    :return: The empty integer interval.
    """
    return IntInterval()


def interval_datetime(
    lower: datetime.datetime, upper: datetime.datetime
) -> DateTimeInterval:
    """
    Creates a new datetime interval.

    :param lower: The lower bound.
    :param upper: The upper bound.
    :return: The new datetime interval.
    """
    return DateTimeInterval.from_atomic(CLOSED, lower, upper, CLOSED)


def empty_interval_datetime() -> DateTimeInterval:
    """
    Creates an empty datetime interval.

    :return: The empty datetime interval.
    """
    return DateTimeInterval()
