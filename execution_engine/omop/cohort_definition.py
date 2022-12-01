import logging
import re
from typing import Iterator

from ..execution_map import ExecutionMap
from .criterion.abstract import AbstractCriterion, Criterion
from .criterion.combination import CriterionCombination


class CohortDefinition:
    """
    A cohort definition in OMOP as a collection of separate criteria
    """

    _criteria: CriterionCombination

    def __init__(self, base_criterion: Criterion) -> None:
        self._base_criterion = base_criterion
        self._criteria = CriterionCombination(
            "root", False, CriterionCombination.Operator("AND")
        )

    def execution_map(self) -> ExecutionMap:
        """
        Get the execution map for the cohort definition.
        """
        return ExecutionMap(self._criteria)

    def add(self, criterion: AbstractCriterion) -> None:
        """
        Add a criterion to the cohort definition.
        """
        self._criteria.add(criterion)

    @staticmethod
    def _to_tablename(name: str, temporary: bool = True) -> str:
        """
        Convert a name to a valid SQL table name.
        """
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        if temporary:
            return f"#{name}"

        return name

    def process(self) -> Iterator[str]:
        """
        Process the cohort definition into a single SQL statement.
        """
        execution_map = self.execution_map()

        i: int
        criterion: Criterion

        table_out = self._to_tablename(f"{self._base_criterion.name}")
        yield self._base_criterion.sql_generate(table_in=None, table_out=table_out)

        for i, criterion in enumerate(execution_map.sequential()):
            table_out = self._to_tablename(f"{criterion.name}_{i}")

            logging.info(
                f"Processing {criterion.name} (exclude={criterion.exclude}) into {table_out}"
            )

            sql = criterion.sql_generate(self._base_criterion.table_out, table_out)

            yield sql

        logging.info("Yielding combination sql")
        sql_template, criteria = execution_map.combine()
        yield sql_template.format(*[c.sql_select() for c in criteria])

        logging.info("Cleaning up temporary tables")
        for criterion in execution_map.sequential():
            logging.info(f"Cleaning up {criterion.name}")
            yield criterion.sql_cleanup()

        yield self._base_criterion.sql_cleanup()
