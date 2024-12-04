from typing import Any, Dict, Iterator, Union

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination


class LogicalCriterionCombination(CriterionCombination):
    """
    A combination of criteria.
    """

    class Operator(CriterionCombination.Operator):
        """Operators for criterion combinations."""

        NOT = "NOT"
        AND = "AND"
        OR = "OR"
        AT_LEAST = "AT_LEAST"
        AT_MOST = "AT_MOST"
        EXACTLY = "EXACTLY"
        ALL_OR_NONE = "ALL_OR_NONE"

        def __init__(self, operator: str, threshold: int | None = None):
            assert operator in [
                "NOT",
                "AND",
                "OR",
                "AT_LEAST",
                "AT_MOST",
                "EXACTLY",
                "ALL_OR_NONE",
            ], f"Invalid operator {operator}"

            self.operator = operator
            if operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
                assert (
                    threshold is not None
                ), f"Threshold must be set for operator {operator}"
            self.threshold = threshold

    @classmethod
    def Not(
        cls,
        criterion: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create a NOT "combination" of a single criterion.
        """
        return cls(
            operator=cls.Operator(cls.Operator.NOT),
            category=category,
            criteria=[criterion],
        )

    @classmethod
    def And(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an AND combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AND),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def Or(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an OR combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.OR),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def AtLeast(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        threshold: int,
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_LEAST, threshold=threshold),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def AtMost(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        threshold: int,
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an AT_MOST combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.AT_MOST, threshold=threshold),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def Exactly(
        cls,
        *criteria: Union[Criterion, "LogicalCriterionCombination"],
        threshold: int,
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an EXACTLY combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.EXACTLY, threshold=threshold),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def AllOrNone(
        cls,
        *criteria: Union[Criterion, "LogicalCriterionCombination"],
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create an ALL_OR_NONE combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.ALL_OR_NONE),
            category=category,
            criteria=criteria,
        )


class NonCommutativeLogicalCriterionCombination(LogicalCriterionCombination):
    """
    A combination of criteria that is not commutative.
    """

    _left: Union[Criterion, CriterionCombination]
    _right: Union[Criterion, CriterionCombination]

    class Operator(CriterionCombination.Operator):
        """Operators for criterion combinations."""

        CONDITIONAL_FILTER = "CONDITIONAL_FILTER"

        def __init__(self, operator: str, threshold: None = None):
            assert operator in [
                "CONDITIONAL_FILTER",
            ], f"Invalid operator {operator}"
            assert threshold is None
            self.operator = operator
            self.threshold = threshold

    def __init__(
        self,
        operator: "NonCommutativeLogicalCriterionCombination.Operator",
        category: CohortCategory,
        left: Union[Criterion, CriterionCombination] | None = None,
        right: Union[Criterion, CriterionCombination] | None = None,
        root_combination: bool = False,
    ):
        """
        Initialize the criterion combination.
        """
        super().__init__(operator=operator, category=category)

        self._criteria = []
        if left is not None:
            self._left = left
        if right is not None:
            self._right = right

        self._root = root_combination

    @property
    def left(self) -> Union[Criterion, CriterionCombination]:
        """
        Get the left criterion.
        """
        return self._left

    @property
    def right(self) -> Union[Criterion, CriterionCombination]:
        """
        Get the right criterion.
        """
        return self._right

    def __str__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        return f"{self.operator}({', '.join(str(c) for c in self._criteria)})"

    def __repr__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        return f'{self.__class__.__name__}("{self.operator}", {self._criteria})'

    def __eq__(self, other: object) -> bool:
        """
        Check if the criterion combination is equal to another criterion combination.
        """
        if not isinstance(other, NonCommutativeLogicalCriterionCombination):
            return NotImplemented
        return (
            self.operator == other.operator
            and self._left == other._left
            and self._right == other._right
            and self.category == other.category
        )

    def __iter__(self) -> Iterator[Union[Criterion, "CriterionCombination"]]:
        """
        Iterate over the criteria in the combination.
        """
        yield self._left
        yield self._right

    def dict(self) -> dict:
        """
        Get the dictionary representation of the criterion combination.
        """
        left = self._left
        right = self._right
        return {
            "operator": self._operator.operator,
            "category": self.category.value,
            "left": {"class_name": left.__class__.__name__, "data": left.dict()},
            "right": {"class_name": right.__class__.__name__, "data": right.dict()},
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "NonCommutativeLogicalCriterionCombination":
        """
        Create a criterion combination from a dictionary.
        """

        from execution_engine.omop.criterion.factory import (
            criterion_factory,  # needs to be here to avoid circular import
        )

        return cls(
            operator=cls.Operator(data["operator"]),
            category=CohortCategory(data["category"]),
            left=criterion_factory(**data["left"]),
            right=criterion_factory(**data["right"]),
        )

    @classmethod
    def ConditionalFilter(
        cls,
        left: Union[Criterion, "CriterionCombination"],
        right: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
    ) -> "LogicalCriterionCombination":
        """
        Create a CONDITIONAL_FILTER combination of criteria.
        """
        return cls(
            operator=cls.Operator(cls.Operator.CONDITIONAL_FILTER),
            category=category,
            left=left,
            right=right,
        )
