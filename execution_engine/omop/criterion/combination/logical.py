from typing import Union

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination


class LogicalCriterionCombination(CriterionCombination):
    """
    A combination of criteria.
    """

    class Operator(CriterionCombination.Operator):
        """Operators for criterion combinations."""

        AND = "AND"
        OR = "OR"
        AT_LEAST = "AT_LEAST"
        AT_MOST = "AT_MOST"
        EXACTLY = "EXACTLY"
        ALL_OR_NONE = "ALL_OR_NONE"

        def __init__(self, operator: str, threshold: int | None = None):
            assert operator in [
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
    def And(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an AND combination of criteria.
        """
        return cls(
            exclude=exclude,
            operator=cls.Operator(cls.Operator.AND),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def Or(
        cls,
        *criteria: Union[Criterion, "CriterionCombination"],
        category: CohortCategory,
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an OR combination of criteria.
        """
        return cls(
            exclude=exclude,
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
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an AT_LEAST combination of criteria.
        """
        return cls(
            exclude=exclude,
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
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an AT_MOST combination of criteria.
        """
        return cls(
            exclude=exclude,
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
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an EXACTLY combination of criteria.
        """
        return cls(
            exclude=exclude,
            operator=cls.Operator(cls.Operator.EXACTLY, threshold=threshold),
            category=category,
            criteria=criteria,
        )

    @classmethod
    def AllOrNone(
        cls,
        *criteria: Union[Criterion, "LogicalCriterionCombination"],
        category: CohortCategory,
        exclude: bool = False,
    ) -> "LogicalCriterionCombination":
        """
        Create an ALL_OR_NONE combination of criteria.
        """
        return cls(
            exclude=exclude,
            operator=cls.Operator(cls.Operator.ALL_OR_NONE),
            category=category,
            criteria=criteria,
        )
