from typing import Any, Dict, Iterator, Union

from ...constants import CohortCategory
from .abstract import AbstractCriterion, Criterion


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

        def __init__(self, operator: str, threshold: int | None = None):
            self.operator = operator
            if operator in ["AT_LEAST", "AT_MOST", "EXACTLY"]:
                assert (
                    threshold is not None
                ), f"Threshold must be set for operator {operator}"
            self.threshold = threshold

    def __init__(
        self, name: str, exclude: bool, operator: Operator, category: CohortCategory
    ):
        """
        Initialize the criterion combination.
        """
        super().__init__(name=name, exclude=exclude, category=category)
        self._operator = operator
        self._criteria: list[Criterion | CriterionCombination] = []

    def add(self, criterion: Criterion | "CriterionCombination") -> None:
        """
        Add a criterion to the combination.
        """
        self._criteria.append(criterion)

    @property
    def name(self) -> str:
        """
        Get the name of the criterion combination.
        """
        return f"CriterionCombination({self.operator.operator}).{self.category}.{self._name}"

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

    def __getitem__(self, item: int) -> Union[Criterion, "CriterionCombination"]:
        """
        Get the criterion at the specified index.
        """
        return self._criteria[item]

    def __repr__(self) -> str:
        """
        Get the string representation of the criterion combination.
        """
        return str(self)

    def dict(self) -> dict:
        """
        Get the dictionary representation of the criterion combination.
        """
        return {
            "name": self.name,
            "exclude": self.exclude,
            "operator": self._operator.operator,
            "threshold": self._operator.threshold,
            "category": self.category.value,
            "criteria": [
                {"class": criterion.__class__.__name__, "data": criterion.dict()}
                for criterion in self._criteria
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CriterionCombination":
        """
        Create a criterion combination from a dictionary.
        """
        operator = cls.Operator(data["operator"], data["threshold"])
        combination = cls(
            name=data["name"],
            exclude=data["exclude"],
            operator=operator,
            category=data["category"],
        )
        # FIXME: Use correct criterion class !
        for criterion in data["criteria"]:
            if criterion["class"] == "CriterionCombination":
                combination.add(CriterionCombination.from_dict(criterion["data"]))
            else:
                combination.add(Criterion.from_dict(criterion["data"]))

        return combination
