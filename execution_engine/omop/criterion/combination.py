from typing import Iterator, Union

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

    def __init__(self, name: str, exclude: bool, operator: Operator, category: str):
        """
        Initialize the criterion combination.
        """
        super().__init__(name=name, exclude=exclude, category=category)
        self._operator = operator
        self._criteria: list[AbstractCriterion] = []

    def add(self, criterion: AbstractCriterion) -> None:
        """
        Add a criterion to the combination.
        """
        self._criteria.append(criterion)

    @property
    def name(self) -> str:
        """
        Get the name of the criterion combination.
        """
        return str(self)

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
