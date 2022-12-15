import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterator

import sqlalchemy
from sqlalchemy import literal_column, select, table, union
from sqlalchemy.orm import Query
from sqlalchemy.sql import Alias, CompoundSelect, Join, Select

from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.util.sql import SelectInto, select_into


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

    def __init__(
        self, base_criterion: Criterion, criteria: CriterionCombination | None = None
    ) -> None:
        self._base_criterion = base_criterion

        if criteria is None:
            self._criteria = CriterionCombination(
                name="root",
                exclude=False,
                category=CohortCategory.BASE,
                operator=CriterionCombination.Operator("AND"),
            )
        else:
            assert isinstance(
                criteria, CriterionCombination
            ), f"Invalid criteria - expected CriterionCombination, got {type(criteria)}"
            assert (
                criteria.category == CohortCategory.BASE
            ), f"Invalid criteria - expected category {CohortCategory.BASE}, got {criteria.category}"

            self._criteria = criteria

        self._execution_map: ExecutionMap | None = None

    def __iter__(self) -> Iterator[Criterion | CriterionCombination]:
        """
        Iterate over the criteria in the cohort definition.
        """
        return iter(self._criteria)

    def __len__(self) -> int:
        """
        Get the number of criteria in the cohort definition.
        """
        return len(self._criteria)

    def __getitem__(self, item: int) -> Criterion | CriterionCombination:
        """
        Get a criterion by index.
        """
        return self._criteria[item]

    def execution_map(self) -> ExecutionMap:
        """
        Get the execution map for the cohort definition.
        """
        return ExecutionMap(self._criteria)

    def add(self, criterion: Criterion | CriterionCombination) -> None:
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

    @staticmethod
    def _assert_base_table_in_select(
        sql: CompoundSelect | Select | SelectInto, base_table_out: str
    ) -> None:
        """
        Assert that the base table is used in the select statement.

        Joining the base table ensures that always just a subset of potients are selected,
        not all.
        """
        if isinstance(sql, SelectInto):
            sql = sql.select

        def _base_table_in_select(sql_select: Join | Select | Alias) -> bool:
            """
            Check if the base table is used in the select statement.
            """
            if isinstance(sql_select, Join):
                return _base_table_in_select(sql_select.left) or _base_table_in_select(
                    sql_select.right
                )
            elif isinstance(sql_select, Select):
                return any(_base_table_in_select(f) for f in sql_select.froms)
            elif isinstance(sql_select, Alias):
                return sql_select.original.concept_name == base_table_out
            else:
                raise NotImplementedError(f"Unknown type {type(sql_select)}")

        if isinstance(sql, CompoundSelect):
            raise NotImplementedError("CompoundSelect not supported")
        elif isinstance(sql, Select):
            assert _base_table_in_select(
                sql
            ), f"Base table {base_table_out} not found in select"
        else:
            raise NotImplementedError(f"Unknown type {type(sql)}")

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

        base_table_out = self._to_tablename(
            f"{table_prefix}{self._base_criterion.name}"
        )
        sql = self._base_criterion.sql_generate(
            table_in=None,
            table_out=base_table_out,
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

            self._assert_base_table_in_select(sql, base_table_out)

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
        return self._execution_map.combine()

    def cleanup(self) -> Iterator[Query]:
        """
        Cleanup the temporary tables.
        """
        if self._execution_map is None:
            raise Exception("Execution map not initialized - run process() first")

        logging.info("Cleaning up temporary tables")
        for criterion in self._execution_map.sequential():
            logging.info(f"Cleaning up {criterion.table_out.original.concept_name}")
            yield to_sqlalchemy(criterion.sql_cleanup())

        yield to_sqlalchemy(self._base_criterion.sql_cleanup())

        self._execution_map = None

    def dict(self) -> dict[str, Any]:
        """
        Get a dictionary representation of the cohort definition.
        """
        return {
            "base_criterion": self._base_criterion.dict(),
            "criteria": self._criteria.dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohortDefinition":
        """
        Create a cohort definition from a dictionary.
        """

        base_criterion = criterion_factory(**data["base_criterion"])
        assert isinstance(
            base_criterion, Criterion
        ), "Base criterion must be a Criterion"

        return cls(
            base_criterion=base_criterion,
            criteria=CriterionCombination.from_dict(data["criteria"]),
        )


class CohortDefinitionCombination:
    """
    A cohort definition combination in OMOP as a collection of separate cohort definitions
    """

    _cohort_definitions: list[CohortDefinition]
    _id: int | None  # The id is used in the cohort_definition_id field in the result tables.
    _recommendation_url: str
    _recommendation_version: str

    def __init__(
        self,
        cohort_definitions: list[CohortDefinition],
        url: str,
        version: str,
        cohort_definition_id: int | None = None,
    ) -> None:
        self._cohort_definitions = cohort_definitions
        self._url = url
        self._version = version
        self._id = cohort_definition_id

    @property
    def id(self) -> int:
        """
        Get the id of the cohort definition combination.

        The id is used in the cohort_definition_id field in the result tables.
        """
        if self._id is None:
            raise Exception("Id not set")
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """
        Set the id of the cohort definition.

        The id is used in the cohort_definition_id field in the result tables.
        """
        self._id = value

    @property
    def url(self) -> str:
        """
        Get the url of the cohort definition combination.
        """
        return self._url

    @property
    def version(self) -> str:
        """
        Get the version of the cohort definition combination.
        """
        return self._version

    def __iter__(self) -> Iterator[CohortDefinition]:
        """
        Iterate over the cohort definitions.
        """
        yield from self._cohort_definitions

    def __len__(self) -> int:
        """
        Get the number of cohort definitions.
        """
        return len(self._cohort_definitions)

    def __getitem__(self, index: int) -> CohortDefinition:
        """
        Get the cohort definition at the given index.
        """
        return self._cohort_definitions[index]

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
            for combination_table in combination_tables:
                assert (
                    len(combination_table.froms) == 1
                ), "Unexpected number of froms in combination table"
                yield to_sqlalchemy(f'DROP TABLE "{str(combination_table.froms[0])}"')
            for cohort_definition in self._cohort_definitions:
                yield from cohort_definition.cleanup()

    def dict(self) -> dict:
        """
        Get the combination as a dictionary.
        """
        return {
            "id": self._id,
            "cohort_definitions": [c.dict() for c in self._cohort_definitions],
            "recommendation_url": self._url,
            "recommendation_version": self._version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CohortDefinitionCombination":
        """
        Create a combination from a dictionary.
        """
        return cls(
            cohort_definitions=[
                CohortDefinition.from_dict(c) for c in data["cohort_definitions"]
            ],
            url=data["recommendation_url"],
            version=data["recommendation_version"],
            cohort_definition_id=data["id"],
        )
