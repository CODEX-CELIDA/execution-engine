import logging
import re
from datetime import datetime
from typing import Any, Iterator

import sqlalchemy
from sqlalchemy import Integer, literal_column, select, table, text, union
from sqlalchemy.orm import Query

from ..execution_map import ExecutionMap
from ..util.sql import select_into
from .criterion.abstract import AbstractCriterion, Criterion
from .criterion.combination import CriterionCombination


def to_sqlalchemy(sql: Any) -> Query:
    """
    Convert a SQL statement to a SQLAlchemy expression.
    """
    if isinstance(sql, str):
        return sqlalchemy.text(sql)
    return sql


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
        self._execution_map: ExecutionMap | None = None

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
        combine: bool = True,
        table_prefix: str = "",
    ) -> Iterator[Query]:
        """
        Process the cohort definition into SQL statements.
        """

        if datetime_end is None:
            datetime_end = datetime.now()

        date_format = "%Y-%m-%d %H:%M:%S"

        logging.info(
            f"Observation window from {datetime_start.strftime(date_format)} to {datetime_end.strftime(date_format)}"
        )

        self._execution_map = self.execution_map()

        i: int
        criterion: Criterion

        table_out = self._to_tablename(f"{table_prefix}{self._base_criterion.name}")
        sql = self._base_criterion.sql_generate(
            table_in=None,
            table_out=table_out,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
        )
        yield to_sqlalchemy(sql)

        for i, criterion in enumerate(self._execution_map.sequential()):
            table_out = self._to_tablename(f"{table_prefix}{criterion.name}_{i}")

            logging.info(
                f"Processing {criterion.name} (exclude={criterion.exclude}) into {table_out}"
            )

            sql = criterion.sql_generate(
                self._base_criterion.table_out, table_out, datetime_start, datetime_end
            )
            # fixme: remove (used for debugging only)
            str(sql)
            yield to_sqlalchemy(sql)

        if combine:
            yield from self.combine()

        if cleanup:
            yield from self.cleanup()

    def combine(self) -> Query:
        """
        Combine the criteria into a single table.
        """
        if self._execution_map is None:
            raise Exception("Execution map not initialized - run process() first")

        logging.info("Yielding combination sql")
        sql_template, criteria = self._execution_map.combine()

        return to_sqlalchemy(sql_template.format(*[c.sql_select() for c in criteria]))

    def cleanup(self) -> Iterator[Query]:
        """
        Cleanup the temporary tables.
        """
        if self._execution_map is None:
            raise Exception("Execution map not initialized - run process() first")

        logging.info("Cleaning up temporary tables")
        for criterion in self._execution_map.sequential():
            logging.info(f"Cleaning up {criterion.table_out.original.name}")
            yield to_sqlalchemy(criterion.sql_cleanup())

        yield to_sqlalchemy(self._base_criterion.sql_cleanup())

        self._execution_map = None


class CohortDefinitionCombination:
    """
    A cohort definition combination in OMOP as a collection of separate cohort definitions
    """

    _cohort_definitions: list[CohortDefinition]

    def __init__(self, cohort_definitions: list[CohortDefinition]) -> None:
        self._cohort_definitions = cohort_definitions

    def process(
        self,
        table_output: str,
        datetime_start: datetime,
        datetime_end: datetime | None,
        table_output_temporary: bool = True,
        cleanup: bool = True,
    ) -> Iterator[Query]:
        """
        Process the cohort definition combination into SQL statements.
        """

        combination_tables = []

        for i, cohort_definition in enumerate(self._cohort_definitions):

            yield from cohort_definition.process(
                datetime_start,
                datetime_end,
                combine=False,
                cleanup=False,
                table_prefix=f"cd{i}_",
            )

            stmt = cohort_definition.combine()
            stmt = stmt.columns(person_id=Integer)
            stmt = stmt.alias(f"cd{i}")
            table_cd = table(f"cd{i}_combined", literal_column("person_id"))
            stmt_into = select_into(select(stmt.c.person_id), table_cd, temporary=True)

            yield to_sqlalchemy(stmt_into)

            combination_tables.append(select(table_cd.c.person_id))

        # expectation is that at least one recommendation-plan must be fulfilled (OR combination)
        stmt = union(*combination_tables)
        stmt_into = select_into(
            stmt,
            table(table_output, literal_column("person_id")),
            temporary=table_output_temporary,
        )

        yield stmt_into

        if cleanup:
            for cohort_definition in self._cohort_definitions:
                yield from cohort_definition.cleanup()
