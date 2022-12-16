import json
import logging
import re

pass
from typing import Any, Dict, Iterator

import sqlalchemy
from sqlalchemy import and_, bindparam, literal_column, select, table, union
from sqlalchemy.orm import Query
from sqlalchemy.sql import (
    Alias,
    CompoundSelect,
    Insert,
    Join,
    Select,
    Subquery,
    TableClause,
)

from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.db.result import RecommendationResult
from execution_engine.util.sql import SelectInto


def to_sqlalchemy(sql: Any) -> Query:
    """
    Convert a SQL statement to a SQLAlchemy expression.
    """
    if isinstance(sql, str):
        return sqlalchemy.text(sql)
    return sql


def add_result_insert(
    query: Select | CompoundSelect,
    name: str | None,
    cohort_category: CohortCategory,
    criterion_name: str | None,
) -> Insert:
    """
    Insert the result of the query into the result table.
    """
    if not isinstance(query, Select) and not isinstance(query, CompoundSelect):
        raise ValueError("sql must be a Select or CompoundSelect")

    query_select: Select

    if isinstance(query, CompoundSelect):
        # CompoundSelect requires the same number of columns for each select, so we need
        # to create a new select with the CompoundSelect as a subquery.
        query_select = select(literal_column("person_id")).select_from(query)
    else:
        query_select = query

    query_select = query_select.add_columns(
        bindparam("run_id").label("recommendation_run_id"),
        bindparam("recommendation_plan_name", name).label("recommendation_plan_name"),
        bindparam("cohort_category", cohort_category.name).label("cohort_category"),
        bindparam("criterion_name", criterion_name).label("criterion_name"),
        # bindparam("criterion_combination_name", self.get_criterion_combination_name(criterion)).label("criterion_combination_name"),
    )

    t_result = RecommendationResult.__table__
    query_insert = t_result.insert().from_select(query_select.columns, query_select)

    return query_insert


class CohortDefinition:
    """
    A cohort definition in OMOP as a collection of separate criteria
    """

    _criteria: CriterionCombination

    def __init__(
        self,
        name: str,
        base_criterion: Criterion,
        criteria: CriterionCombination | None = None,
    ) -> None:
        self._name = name
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

    @property
    def name(self) -> str:
        """
        Get the name of the cohort definition.
        """
        return self._name

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
    def _to_table(name: str) -> TableClause:
        """
        Convert a name to a valid SQL table name.
        """
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        return table(name, literal_column("person_id"))

    @staticmethod
    def _assert_base_table_in_select(
        sql: CompoundSelect | Select | SelectInto, base_table_out: str
    ) -> None:
        """
        Assert that the base table is used in the select statement.

        Joining the base table ensures that always just a subset of potients are selected,
        not all.
        """
        if isinstance(sql, SelectInto) or isinstance(sql, Insert):
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
                return sql_select.original.name == base_table_out
            elif isinstance(sql_select, TableClause):
                return sql_select.name == base_table_out
            elif isinstance(sql_select, Subquery):
                return any(_base_table_in_select(f) for f in sql_select.original.froms)
            else:
                raise NotImplementedError(f"Unknown type {type(sql_select)}")

        if isinstance(sql, CompoundSelect):
            assert all(
                _base_table_in_select(s) for s in sql.selects
            ), "Base table not used in all selects of compound select"
        elif isinstance(sql, Select):
            assert _base_table_in_select(
                sql
            ), f"Base table {base_table_out} not found in select"
        else:
            raise NotImplementedError(f"Unknown type {type(sql)}")

    def process(
        self,
        table_prefix: str = "",
    ) -> Iterator[Query]:
        """
        Process the cohort definition into SQL statements.
        """

        self._execution_map = self.execution_map()

        i: int
        criterion: Criterion

        base_table = self._to_table(f"{table_prefix}{self._base_criterion.name}")
        query = self._base_criterion.sql_generate(base_table=base_table)
        query = self._base_criterion.sql_insert_into_table(
            query, base_table, temporary=True
        )
        assert isinstance(
            query, SelectInto
        ), f"Invalid query - expected SelectInto, got {type(query)}"

        yield to_sqlalchemy(query)

        for i, criterion in enumerate(self._execution_map.sequential()):
            # table_out = self._to_table(f"{table_prefix}{criterion.name}_{i}")

            logging.info(f"Processing {criterion.name} (exclude={criterion.exclude})")

            sql = criterion.sql_generate(base_table=base_table)
            # fixme: remove (used for debugging only)
            str(sql)

            self._assert_base_table_in_select(sql, base_table.name)
            sql = add_result_insert(
                sql,
                name=self.name,
                cohort_category=criterion.category,
                criterion_name=criterion.unique_name(),
            )

            yield to_sqlalchemy(sql)

        yield from self.combine()

    def get_criterion_combination_name(self, criterion: Criterion) -> str:
        """
        Get the name of the criterion combination for the given criterion.
        """

        def _traverse(comb: CriterionCombination) -> str:
            for element in comb:
                if isinstance(element, CriterionCombination):
                    yield from _traverse(element)
                else:
                    yield comb, element

        comb: CriterionCombination
        element: Criterion

        for comb, element in _traverse(self._criteria):
            if element.dict() == criterion.dict():
                return comb.unique_name()

        raise ValueError(f"Criterion {criterion.name} not found in cohort definition")

    def combine(self) -> Query:
        """
        Combine the criteria into a single table.
        """
        if self._execution_map is None:
            raise Exception("Execution map not initialized - run process() first")

        logging.info("Yielding combination statements")

        for category in [
            CohortCategory.POPULATION,
            CohortCategory.INTERVENTION,
            CohortCategory.POPULATION_INTERVENTION,
        ]:
            query = self._execution_map.combine(cohort_category=category)
            query = add_result_insert(
                query,
                name=self.name,
                cohort_category=category,
                criterion_name=category.name,
            )
            yield query

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
            name=data["name"],
            base_criterion=base_criterion,
            criteria=CriterionCombination.from_dict(data["criteria"]),
        )


class CohortDefinitionCombination:
    """
    A cohort definition combination in OMOP as a collection of separate cohort definitions
    """

    def __init__(
        self,
        cohort_definitions: list[CohortDefinition],
        url: str,
        version: str,
        cohort_definition_id: int | None = None,
    ) -> None:
        self._cohort_definitions: list[CohortDefinition] = cohort_definitions
        self._url: str = url
        self._version: str = version
        # The id is used in the cohort_definition_id field in the result tables.
        self._id: int | None = cohort_definition_id

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

    def process(self) -> Iterator[Query]:
        """
        Process the cohort definition combination into SQL statements.
        """

        for i, cohort_definition in enumerate(self._cohort_definitions):

            yield from cohort_definition.process(
                table_prefix=f"cd{i}_",
            )

        yield from self.combine()

    def combine(self) -> Iterator[Insert]:
        """
        Combine the results of the individual cohort definitions.
        """

        table = RecommendationResult.__table__

        def _get_query(
            cohort_definition: CohortDefinition, category: CohortCategory
        ) -> Select:
            return select(table.c.person_id).where(
                and_(
                    table.c.recommendation_plan_name == cohort_definition.name,
                    table.c.cohort_category == category.name,
                    table.c.recommendation_run_id == bindparam("run_id"),
                )
            )

        def get_statements(cohort_category: CohortCategory) -> list[Select]:
            return [_get_query(cd, cohort_category) for cd in self._cohort_definitions]

        for category in [
            CohortCategory.POPULATION,
            CohortCategory.INTERVENTION,
            CohortCategory.POPULATION_INTERVENTION,
        ]:
            statements = get_statements(category)
            query = union(*statements)
            query = add_result_insert(
                query, name=None, cohort_category=category, criterion_name=None
            )

            yield query

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

    @classmethod
    def from_json(cls, data: str) -> "CohortDefinitionCombination":
        """
        Create a combination from a JSON string.
        """
        return cls.from_dict(json.loads(data))
