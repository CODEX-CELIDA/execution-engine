from portion import CLOSED, AbstractDiscreteInterval


class IntInterval(AbstractDiscreteInterval):
    """
    An integer interval.
    """

    _step = 1


def interval(lower: int, upper: int) -> IntInterval:
    """
    Creates a new integer interval.

    :param lower: The lower bound.
    :param upper: The upper bound.
    :return: The new integer interval.
    """
    return IntInterval.from_atomic(CLOSED, lower, upper, CLOSED)


def empty_interval() -> IntInterval:
    """
    Creates an empty integer interval.

    :return: The empty integer interval.
    """
    return IntInterval()
