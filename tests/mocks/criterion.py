from sqlalchemy import Select

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion


class MockCriterion(Criterion):
    def _create_query(self) -> Select:
        pass

    def __init__(
        self,
        name: str,
    ):
        self._id = None
        self._name = name

    def unique_name(self) -> str:
        return self._name

    def _sql_generate(self, query: Select) -> Select:
        pass

    def _sql_filter_concept(self, query: Select) -> Select:
        pass

    def _sql_select_data(self, query: Select) -> Select:
        pass

    def description(self) -> str:
        return f"MockCriterion[{self._name}]"

    @property
    def concept(self) -> Concept:  # type: ignore
        pass
