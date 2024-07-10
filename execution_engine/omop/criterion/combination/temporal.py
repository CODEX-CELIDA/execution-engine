from datetime import time
from enum import StrEnum
from typing import Any, Dict, Union

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination


class TimeIntervalType(StrEnum):
    """
    Types of time intervals to aggregate criteria over.
    """

    MORNING_SHIFT = "morning_shift"
    AFTERNOON_SHIFT = "afternoon_shift"
    NIGHT_SHIFT = "night_shift"
    DAY = "day"


class TemporalIndicatorCombination(CriterionCombination):
    """
    A combination of criteria.
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
        category: CohortCategory,
        criterion: Union[Criterion, "CriterionCombination"] | None = None,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ):
        if criterion is not None:
            if not isinstance(criterion, (Criterion, CriterionCombination)):
                raise ValueError(
                    f"Invalid criterion - expected Criterion or CriterionCombination, got {type(criterion)}"
                )
            criteria = [criterion]
        else:
            criteria = None

        super().__init__(
            exclude=False, operator=operator, category=category, criteria=criteria
        )

        if interval_type:
            if start_time is not None or end_time is not None:
                raise ValueError(
                    "start_time and end_time must not be provided if an interval type is given"
                )
            if interval_type not in TimeIntervalType:
                raise ValueError(f"Invalid interval type: {interval_type}")
        else:
            if start_time is None or end_time is None:
                raise ValueError(
                    "start_time and end_time must be provided if no interval type is given"
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

    def dict(self) -> Dict:
        """
        Get the dictionary representation of the criterion combination.
        """

        d = super().dict()
        del d["exclude"]
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        d["interval_type"] = self.interval_type

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalIndicatorCombination":
        """
        Create a criterion combination from a dictionary.
        """

        from execution_engine.omop.criterion.factory import (
            criterion_factory,  # needs to be here to avoid circular import
        )

        operator = cls.Operator(data["operator"], data["threshold"])
        category = CohortCategory(data["category"])

        combination = cls(
            operator=operator,
            category=category,
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
        category: CohortCategory,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "TemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=1),
            category=category,
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
        category: CohortCategory,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "TemporalIndicatorCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=threshold),
            category=category,
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
        category: CohortCategory,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "TemporalIndicatorCombination":
        """
        Create an AT_MOST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_MOST, threshold=threshold),
            category=category,
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
        category: CohortCategory,
        interval_type: TimeIntervalType | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
    ) -> "TemporalIndicatorCombination":
        """
        Create an EXACTLY combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.EXACTLY, threshold=threshold),
            category=category,
            criterion=criterion,
            interval_type=interval_type,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def MorningShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "TemporalIndicatorCombination":
        """
        Create a MorningShift combination of criteria.
        """

        return cls.Presence(criterion, category, TimeIntervalType.MORNING_SHIFT)

    @classmethod
    def AfternoonShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "TemporalIndicatorCombination":
        """
        Create a AfternoonShift combination of criteria.
        """

        return cls.Presence(criterion, category, TimeIntervalType.AFTERNOON_SHIFT)

    @classmethod
    def NightShift(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "TemporalIndicatorCombination":
        """
        Create a NightShift combination of criteria.
        """

        return cls.Presence(criterion, category, TimeIntervalType.NIGHT_SHIFT)

    @classmethod
    def Day(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "TemporalIndicatorCombination":
        """
        Create a Day combination of criteria.
        """

        return cls.Presence(criterion, category, TimeIntervalType.DAY)
