from abc import ABC
from datetime import time
from enum import StrEnum
from typing import Any, Dict, Iterator, Union

from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination


class TimeIntervalType(StrEnum):
    """
    Types of time intervals to aggregate criteria over.
    """

    MORNING_SHIFT = "morning_shift"
    AFTERNOON_SHIFT = "afternoon_shift"
    NIGHT_SHIFT = "night_shift"
    NIGHT_SHIFT_BEFORE_MIDNIGHT = "night_shift_before_midnight"
    NIGHT_SHIFT_AFTER_MIDNIGHT = "night_shift_after_midnight"
    DAY = "day"
    ANY_TIME = "any_time"

    def __repr__(self) -> str:
        """
        Get the string representation of the time interval type.
        """
        return f"{self.__class__.__name__}.{self.name}"


class TemporalIndicatorCombination(CriterionCombination, ABC):
    """
    TemporalIndicatorCombination is an abstract base class for constructing temporal indicator
    combinations used to evaluate patient data over time. It encapsulates the common logic such as
    operator definitions (e.g., AT_LEAST, AT_MOST, EXACTLY) and the management of one or more criteria.
    This class serves as the foundation for more specialized implementations that define how the time
    windows for evaluation are determined.
    """

    class Operator(CriterionCombination.Operator):
        """Operators for criterion combinations."""

        AT_LEAST = "AT_LEAST"
        AT_MOST = "AT_MOST"
        EXACTLY = "EXACTLY"

        def __init__(self, operator: str, threshold: int | None = None):
            assert operator in [
                "AT_LEAST",
                "AT_MOST",
                "EXACTLY",
            ], f"Invalid operator {operator}"

            self.operator = operator
            if operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
                assert (
                    threshold is not None
                ), f"Threshold must be set for operator {operator}"
            self.threshold = threshold

    def __init__(
        self,
        operator: Operator,
        criterion: Union[Criterion, "CriterionCombination"] | None = None,
    ):
        if criterion is not None:
            if not isinstance(criterion, (Criterion, CriterionCombination)):
                raise ValueError(
                    f"Invalid criterion - expected Criterion or CriterionCombination, got {type(criterion)}"
                )
            criteria = [criterion]
        else:
            criteria = None

        super().__init__(operator=operator, criteria=criteria)


class FixedWindowTemporalIndicatorCombination(TemporalIndicatorCombination):
    """
    FixedWindowTemporalIndicatorCombination implements a temporal indicator combination that relies on
    fixed time window specifications. It supports two mutually exclusive methods for defining these windows:
    either via a pre-defined TimeIntervalType (e.g., morning, afternoon, or night shifts) or through explicit
    start_time and end_time values. This class is intended for scenarios where the same evaluation window
    applies uniformly across all patients, and it enforces validation to ensure only one method of window
    specification is used.
    """

    interval_type: TimeIntervalType | None = None
    start_time: time | None = None
    end_time: time | None = None

    def __init__(
        self,
        operator: TemporalIndicatorCombination.Operator,
        criterion: Union[Criterion, "CriterionCombination"] | None = None,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ):
        super().__init__(operator=operator, criterion=criterion)

        if interval_type:
            if start_time is not None or end_time is not None:
                raise ValueError(
                    "start_time/end_time cannot be used together with interval_type"
                )
            # Validate the interval_type if needed
            self.interval_type = interval_type
            self.start_time = None
            self.end_time = None
        else:
            # Must have start_time and end_time
            if start_time is None or end_time is None:
                raise ValueError(
                    "Either interval_type OR both start_time & end_time must be provided"
                )
            if start_time >= end_time:
                raise ValueError("start_time must be less than end_time")

        self.interval_type = interval_type
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        if self.interval_type:
            return f"{super().__str__()} for {self.interval_type.value}"
        elif self.start_time and self.end_time:
            return f"{super().__str__()} from {self.start_time.strftime('%H:%M:%S')} to {self.end_time.strftime('%H:%M:%S')}"
        else:
            return super().__str__()

    def _repr_pretty(self, level: int = 0) -> str:
        children = [(None, c) for c in self._criteria]
        params = [
            ("interval_type", self.interval_type),
            ("start_time", self.start_time),
            ("end_time", self.end_time),
        ]
        return self._build_repr(children, params, level)

    def dict(self) -> Dict:
        """
        Get the dictionary representation of the criterion combination.
        """

        d = super().dict()
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        d["interval_type"] = self.interval_type

        return d

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a criterion combination from a dictionary.
        """

        from execution_engine.omop.criterion.factory import (
            criterion_factory,  # needs to be here to avoid circular import
        )

        operator = cls.Operator(data["operator"], data["threshold"])

        combination = cls(
            operator=operator,
            interval_type=data["interval_type"],
            # start_time and end_time is in isoformat !
            start_time=(
                time.fromisoformat(data["start_time"]) if data["start_time"] else None
            ),
            end_time=time.fromisoformat(data["end_time"]) if data["end_time"] else None,
        )

        for criterion in data["criteria"]:
            combination.add(criterion_factory(**criterion))

        return combination

    @classmethod
    def Presence(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=1),
            criterion=criterion,
            interval_type=interval_type,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def MinCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=threshold),
            criterion=criterion,
            interval_type=interval_type,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def MaxCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
        interval_criterion: Criterion | CriterionCombination | None = None,
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create an AT_MOST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_MOST, threshold=threshold),
            criterion=criterion,
            interval_type=interval_type,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def ExactCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create an EXACTLY combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.EXACTLY, threshold=threshold),
            criterion=criterion,
            interval_type=interval_type,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def MorningShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a MorningShift combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.MORNING_SHIFT)

    @classmethod
    def AfternoonShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a AfternoonShift combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.AFTERNOON_SHIFT)

    @classmethod
    def NightShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a NightShift combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.NIGHT_SHIFT)

    @classmethod
    def NightShiftBeforeMidnight(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a NightShiftBeforeMidnight combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.NIGHT_SHIFT_BEFORE_MIDNIGHT)

    @classmethod
    def NightShiftAfterMidnight(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a NightShiftAfterMidnight combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.NIGHT_SHIFT_AFTER_MIDNIGHT)

    @classmethod
    def Day(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a Day combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.DAY)

    @classmethod
    def AnyTime(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
    ) -> "FixedWindowTemporalIndicatorCombination":
        """
        Create a AnyTime combination of criteria.
        """
        return cls.Presence(criterion, TimeIntervalType.ANY_TIME)


class PersonalWindowTemporalIndicatorCombination(TemporalIndicatorCombination):
    """
    PersonalWindowTemporalIndicatorCombination implements a temporal indicator combination based on
    person-specific time windows. Instead of using fixed start/end times or a global TimeIntervalType, this
    class leverages an interval_criterion to dynamically generate evaluation windows tailored to each
    patient. This design is ideal for situations where the timing of events (such as post-operative periods)
    varies between patients, enabling more personalized temporal assessments.
    """

    _interval_criterion: Criterion | CriterionCombination

    def __init__(
        self,
        operator: TemporalIndicatorCombination.Operator,
        criterion: Union[Criterion, "CriterionCombination"] | None,
        interval_criterion: Criterion | CriterionCombination,
    ):
        super().__init__(operator=operator, criterion=criterion)

        if not isinstance(interval_criterion, (Criterion, CriterionCombination)):
            raise ValueError(
                f"Invalid criterion - expected Criterion or CriterionCombination, got {type(interval_criterion)}"
            )

        self._interval_criterion = interval_criterion

    def __iter__(self) -> Iterator[Union[Criterion, "CriterionCombination"]]:
        """
        Iterate over the criteria in the combination - first criteria, then interval criterion if present.
        """
        yield from super().__iter__()
        yield self._interval_criterion

    def __str__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        base_str = super().__str__()
        return f"{base_str} [Personal Windows via: {self.interval_criterion}]"

    def _repr_pretty(self, level: int = 0) -> str:
        children = [(None, c) for c in self._criteria]
        params = [
            ("interval_criterion", self.interval_criterion),
        ]
        return self._build_repr(children, params, level)

    @property
    def interval_criterion(self) -> Criterion | CriterionCombination:
        """
        Get the interval criterion.
        """
        return self._interval_criterion

    def dict(self) -> Dict:
        """
        Get the dictionary representation of the criterion combination.
        """

        d = super().dict()
        d["interval_criterion"] = {
            "class_name": self.interval_criterion.__class__.__name__,
            "data": self.interval_criterion.dict(),
        }

        return d

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "PersonalWindowTemporalIndicatorCombination":
        """
        Create a criterion combination from a dictionary.
        """

        from execution_engine.omop.criterion.factory import (
            criterion_factory,  # needs to be here to avoid circular import
        )

        operator = cls.Operator(data["operator"], data["threshold"])

        combination = cls(
            operator=operator,
            criterion=None,
            interval_criterion=criterion_factory(**data["interval_criterion"]),
        )

        for criterion in data["criteria"]:
            combination.add(criterion_factory(**criterion))

        return combination

    @classmethod
    def Presence(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        interval_criterion: Criterion | CriterionCombination,
    ) -> "PersonalWindowTemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=1),
            criterion=criterion,
            interval_criterion=interval_criterion,
        )

    @classmethod
    def MinCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_criterion: Criterion | CriterionCombination,
    ) -> "TemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=threshold),
            criterion=criterion,
            interval_criterion=interval_criterion,
        )

    @classmethod
    def MaxCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_criterion: Criterion | CriterionCombination,
    ) -> "TemporalIndicatorCombination":
        """
        Create an AT_MOST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_MOST, threshold=threshold),
            criterion=criterion,
            interval_criterion=interval_criterion,
        )

    @classmethod
    def ExactCount(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        threshold: int,
        interval_criterion: Criterion | CriterionCombination,
    ) -> "TemporalIndicatorCombination":
        """
        Create an EXACTLY combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.EXACTLY, threshold=threshold),
            criterion=criterion,
            interval_criterion=interval_criterion,
        )
