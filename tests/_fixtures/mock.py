from typing import Any, Dict, Self

from sqlalchemy import Select

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion


class MockCriterion(Criterion):
    def __init__(
        self,
        name: str,
        exclude: bool = False,
        category: CohortCategory = CohortCategory.POPULATION,
    ):
        self._name = name
        self._exclude = exclude
        self._category = category

    def unique_name(self) -> str:
        return self._name

    def _sql_generate(self, query: Select) -> Select:
        pass

    def _sql_filter_concept(self, query: Select) -> Select:
        pass

    def _sql_select_data(self, query: Select) -> Select:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
            name=data["name"],
            exclude=data["exclude"],
            category=CohortCategory(data["category"]),
        )

    def dict(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "exclude": self._exclude,
            "category": self._category.value,
        }

    def __repr__(self) -> str:
        return f"MockCriterion({self._name})"

    @property
    def concept(self) -> Concept:  # type: ignore
        pass
