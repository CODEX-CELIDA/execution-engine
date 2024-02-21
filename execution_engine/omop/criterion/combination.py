from typing import Any, Dict, Iterator, Sequence, Union, cast

from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import AbstractCriterion, Criterion

__all__ = ["CriterionCombination"]


class CriterionCombination(AbstractCriterion):
    """
    A combination of criteria.
    """

    class Operator:
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
                return f'CriterionCombination.Operator("{self.operator}", threshold={self.threshold})'
            else:
                return f'CriterionCombination.Operator("{self.operator}")'

        def __eq__(self, other: object) -> bool:
            """
            Check if the operator is equal to another operator.
            """
            if not isinstance(other, CriterionCombination.Operator):
                return NotImplemented
            return self.operator == other.operator and self.threshold == other.threshold

    def __init__(
        self,
        name: str,
        exclude: bool,
        operator: Operator,
        category: CohortCategory,
        criteria: Sequence[Union[Criterion, "CriterionCombination"]] | None = None,
    ):
        """
        Initialize the criterion combination.
        """
        super().__init__(name=name, exclude=exclude, category=category)
        self._operator = operator

        self._criteria: list[Union[Criterion, CriterionCombination]]

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

    def add_all(self, criteria: list[Union[Criterion, "CriterionCombination"]]) -> None:
        """
        Add multiple criteria to the combination.
        """
        self._criteria.extend(criteria)

    @property
    def name(self) -> str:
        """
        Get the name of the criterion combination.
        """
        return f"CriterionCombination({self.operator}).{self.category.value}.{self._name}(exclude={self._exclude})"

    @property
    def raw_name(self) -> str:
        """
        Get the name of the criterion combination.
        """
        return self._name

    @property
    def operator(self) -> "CriterionCombination.Operator":
        """
        Get the operator of the criterion combination (i.e. the type of combination, e.g. AND, OR, AT_LEAST, etc.).
        """
        return self._operator

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

    def dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the criterion combination.
        """
        return {
            "name": self._name,
            "exclude": self._exclude,
            "operator": self._operator.operator,
            "threshold": self._operator.threshold,
            "category": self._category.value,
            "criteria": [
                {"class_name": criterion.__class__.__name__, "data": criterion.dict()}
                for criterion in self._criteria
            ],
        }

    def __invert__(self) -> "CriterionCombination":
        """
        Invert the criterion combination.
        """
        return CriterionCombination(
            name=self._name,
            exclude=not self._exclude,
            operator=self._operator,
            category=self._category,
            criteria=self._criteria,
        )

    def invert(self) -> "CriterionCombination":
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
            name=data["name"],
            exclude=data["exclude"],
            operator=operator,
            category=category,
        )

        for criterion in data["criteria"]:
            combination.add(criterion_factory(**criterion))

        return combination
