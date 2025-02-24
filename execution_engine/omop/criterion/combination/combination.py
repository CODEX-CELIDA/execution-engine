from abc import ABCMeta
from typing import Any, Dict, Iterator, Sequence, Union, cast

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import AbstractCriterion, Criterion

__all__ = ["CriterionCombination"]


class CriterionCombination(AbstractCriterion, metaclass=ABCMeta):
    """
    Base class for a combination of criteria (temporal or logical).
    """

    class Operator:
        """
        Operators for criterion combinations.
        """

        def __init__(self, operator: str, threshold: int | None = None):
            self.operator = operator
            self.threshold = threshold

        def __str__(self) -> str:
            """
            Get the string representation of the operator.
            """
            if self.operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
                return f"{self.operator}(threshold={self.threshold})"
            else:
                return f"{self.operator}"

        def __repr__(self) -> str:
            """
            Get the string representation of the operator.
            """
            if self.operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
                return f'{self.__class__.__name__}("{self.operator}", threshold={self.threshold})'
            else:
                return f'{self.__class__.__name__}("{self.operator}")'

        def __eq__(self, other: object) -> bool:
            """
            Check if the operator is equal to another operator.
            """
            if not isinstance(other, CriterionCombination.Operator):
                return NotImplemented
            return self.operator == other.operator and self.threshold == other.threshold

    def __init__(
        self,
        operator: Operator,
        category: CohortCategory,
        criteria: Sequence[Union[Criterion, "CriterionCombination"]] | None = None,
        root_combination: bool = False,
    ):
        """
        Initialize the criterion combination.
        """
        super().__init__(category=category)
        self._operator = operator

        self._criteria: list[Union[Criterion, CriterionCombination]]
        self._root = root_combination

        if criteria is None:
            self._criteria = []
        else:
            self._criteria = cast(
                list[Union[Criterion, "CriterionCombination"]], criteria
            )

    def add(self, criterion: Union[Criterion, "CriterionCombination"]) -> None:
        """
        Add a criterion to the combination.
        """
        self._criteria.append(criterion)

    def add_all(
        self, criteria: Sequence[Union[Criterion, "CriterionCombination"]]
    ) -> None:
        """
        Add multiple criteria to the combination.
        """
        self._criteria.extend(criteria)

    def __str__(self) -> str:
        """
        Get the name of the criterion combination.
        """
        return f"{self.__class__.__name__}({self.operator}).{self.category.value}"

    @property
    def operator(self) -> "CriterionCombination.Operator":
        """
        Get the operator of the criterion combination (i.e. the type of combination, e.g. AND, OR, AT_LEAST, etc.).
        """
        return self._operator

    def set_root(self, value: bool = True) -> None:
        """
        Sets whether this criterion combination is at the root of a tree of criteria / combinations.
        """
        self._root = value

    def is_root(self) -> bool:
        """
        Returns whether this criterion combination is at the root of a tree of criteria / combinations.
        """
        return self._root

    def __iter__(self) -> Iterator[Union[Criterion, "CriterionCombination"]]:
        """
        Iterate over the criteria in the combination.
        """
        for criterion in self._criteria:
            yield criterion

    def __len__(self) -> int:
        """
        Get the number of criteria in the combination.
        """
        return len(self._criteria)

    def __getitem__(self, index: int) -> Union[Criterion, "CriterionCombination"]:
        """
        Get the criterion at the specified index.
        """
        return self._criteria[index]

    def __repr__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        return str(self)

    def description(self) -> str:
        """
        Description of this combination.
        """
        return str(self)

    def dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the criterion combination.
        """
        return {
            "operator": self._operator.operator,
            "threshold": self._operator.threshold,
            "category": self._category.value,
            "criteria": [
                {
                    "class_name": criterion.__class__.__name__,
                    "data": criterion.dict(),
                }
                for criterion in self._criteria
            ],
            "root": self._root,
        }

    def __invert__(self) -> AbstractCriterion:
        """
        Invert the criterion combination.
        """
        # Would be cycle if imported at top-level.
        from execution_engine.omop.criterion.combination.logical import (
            LogicalCriterionCombination,
        )

        if (
            isinstance(self, LogicalCriterionCombination)
            and self.operator.operator == LogicalCriterionCombination.Operator.NOT
        ):
            return self._criteria[0]
        else:
            copy = self.__class__(
                operator=self._operator,
                category=self._category,
                criteria=self._criteria,
            )
            return LogicalCriterionCombination.Not(copy, self._category)

    def invert(self) -> AbstractCriterion:
        """
        Invert the criterion combination.
        """
        return ~self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CriterionCombination":
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
            root_combination=data["root"],
        )

        for criterion in data["criteria"]:
            combination.add(criterion_factory(**criterion))

        return combination
