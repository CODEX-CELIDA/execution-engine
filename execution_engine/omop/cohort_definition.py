import logging
import re
from datetime import datetime
from typing import Any, Iterator

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.orm import Query

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

        return name

    def process(
        self,
        datetime_start: datetime,
        datetime_end: datetime | None,
        cleanup: bool = True,
    ) -> Iterator[Query]:
        """
        Process the cohort definition into SQL statements.
        """

        def to_sqlalchemy(sql: Any) -> Query:
            """
            Convert a SQL statement to a SQLAlchemy expression.
            """
            if isinstance(sql, str):
                return sqlalchemy.text(sql)
            return sql

        if datetime_end is None:
            datetime_end = datetime.now()

        date_format = "%Y-%m-%d %H:%M:%S"

        logging.info(
            f"Observation window from {datetime_start.strftime(date_format)} to {datetime_end.strftime(date_format)}"
        )

        execution_map = self.execution_map()

        i: int
        criterion: Criterion

        table_out = self._to_tablename(f"{self._base_criterion.name}")
        sql = self._base_criterion.sql_generate(
            table_in=None,
            table_out=table_out,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
        )
        yield to_sqlalchemy(sql)

        for i, criterion in enumerate(execution_map.sequential()):
            table_out = self._to_tablename(f"{criterion.name}_{i}")

            logging.info(
                f"Processing {criterion.name} (exclude={criterion.exclude}) into {table_out}"
            )

            sql = criterion.sql_generate(
                self._base_criterion.table_out, table_out, datetime_start, datetime_end
            )
            print(sql)
            yield to_sqlalchemy(sql)

        yield text("COMMIT")

        logging.info("Yielding combination sql")
        sql_template, criteria = execution_map.combine()
        yield to_sqlalchemy(sql_template.format(*[c.sql_select() for c in criteria]))

        yield text("COMMIT")

        if cleanup:
            logging.info("Cleaning up temporary tables")
            for criterion in execution_map.sequential():
                logging.info(f"Cleaning up {criterion.name}")
                yield to_sqlalchemy(criterion.sql_cleanup())

            yield to_sqlalchemy(self._base_criterion.sql_cleanup())

            yield text("COMMIT")
