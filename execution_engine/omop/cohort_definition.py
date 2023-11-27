import itertools
import logging
import re
from typing import Any, Dict, Iterator, Self

from sqlalchemy import (
    Column,
    Date,
    Enum,
    Index,
    Integer,
    MetaData,
    Table,
    and_,
    bindparam,
    select,
    union,
)
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
from sqlalchemy.sql.ddl import DropTable
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.selectable import CTE

from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.db.celida import RecommendationResult
from execution_engine.omop.serializable import Serializable
from execution_engine.util.sql import SelectInto


def add_result_insert(
    query: Select | CompoundSelect,
    plan_id: int | None,
    criterion_id: int | None,
    cohort_category: CohortCategory,
) -> Insert:
    """
    Insert the result of the query into the result table.
    """
    if not isinstance(query, Select) and not isinstance(query, CompoundSelect):
        raise ValueError("sql must be a Select or CompoundSelect")

    query_select: Select

    # Always surround the original query by a select () query, as
    # otherwise problemes arise when using CompoundSelect, multiple columns or DISTINCT person_id
    description = query.description
    query = query.alias("base_select")
    query_select = select(query.c.person_id, query.c.valid_date).select_from(query)

    query_select = query_select.add_columns(
        bindparam("run_id", type_=Integer()).label("recommendation_run_id"),
        bindparam("plan_id", plan_id).label("plan_id"),
        bindparam("cohort_category", cohort_category, type_=Enum(CohortCategory)).label(
            "cohort_category"
        ),
        bindparam("criterion_id", criterion_id).label("criterion_id"),
    )

    t_result = RecommendationResult.__table__
    query_insert = t_result.insert().from_select(
        query_select.selected_columns, query_select
    )

    query_select.description = description
    query_insert.description = description

    return query_insert


class CohortDefinition(Serializable):
    """
    A cohort definition in OMOP as a collection of separate criteria.

    A cohort definition represents an individual recommendation plan (i.e. one part of a single recommendation),
    whereas a cohort definition combination represents the whole recommendation, consisting of one or multiple
    recommendation plans = cohort definitions.
    In turn, a cohort definition is a collection of criteria, which can be either a single criterion or a combination
    of criteria (i.e. a criterion combination).
    These single criteria can be either a single criterion (e.g. "has condition X") or a combination of criteria
    (e.g. "has condition X and lab value Y >= Z").
    """

    _id: int | None = None
    _name: str
    _criteria: CriterionCombination
    _execution_map: ExecutionMap | None = None

    def __init__(
        self,
        name: str,
        url: str,
        base_criterion: Criterion,
        criteria: CriterionCombination | None = None,
    ) -> None:
        self._name = name
        self._url = url
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

    @property
    def url(self) -> str:
        """
        Get the url of the cohort definition combination.
        """
        return self._url

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

    def get_criterion(self, criterion_unique_name: str) -> Criterion | None:
        """
        Retrieve a criterion by its unique name.
        """

        def _traverse(comb: CriterionCombination) -> Criterion | None:
            for element in comb:
                if isinstance(element, CriterionCombination):
                    found = _traverse(element)
                    if found is not None:
                        return found
                elif element.unique_name() == criterion_unique_name:
                    return element
            return None

        return _traverse(self._criteria)

    def criteria(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list
        """

        def _traverse(comb: CriterionCombination) -> list[Criterion]:
            criteria = []
            for element in comb:
                if isinstance(element, CriterionCombination):
                    criteria += _traverse(element)
                else:
                    criteria.append(element)
            return criteria

        return _traverse(self._criteria)

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
                return any(
                    _base_table_in_select(f) for f in sql_select.get_final_froms()
                ) or any(_base_table_in_select(w) for w in sql_select.whereclause)
            elif isinstance(sql_select, Alias):
                return sql_select.original.name == base_table_out
            elif isinstance(sql_select, TableClause):
                return sql_select.name == base_table_out
            elif isinstance(sql_select, CTE):
                return _base_table_in_select(sql_select.original)
            elif isinstance(sql_select, BooleanClauseList):
                return any(
                    w.right.element.froms[0].name == base_table_out for w in sql_select
                )
            elif isinstance(sql_select, BinaryExpression):
                return (
                    sql_select.right.element.get_final_froms()[0].name == base_table_out
                )
            elif isinstance(sql_select, Subquery):
                if isinstance(sql_select.original, CompoundSelect):
                    return all(
                        _base_table_in_select(s) for s in sql_select.original.selects
                    )
                else:
                    return any(
                        _base_table_in_select(f)
                        for f in sql_select.original.get_final_froms()
                    )
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

    def execution_plan(self) -> Iterator[Criterion]:
        yield from self.execution_map().sequential()

    def process(self, base_table: Table) -> Iterator[Query]:
        """
        Process the cohort definition into SQL statements.
        """

        self._execution_map = self.execution_map()

        i: int
        criterion: Criterion

        for i, criterion in enumerate(self._execution_map.sequential()):
            logging.info(f"Processing {criterion.description()}")

            query = criterion.sql_generate(base_table=base_table)
            self._assert_base_table_in_select(query, base_table.name)

            query = add_result_insert(
                query,
                plan_id=self.id,
                criterion_id=criterion.id,
                cohort_category=criterion.category,
            )

            yield query

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
                plan_id=self.id,
                criterion_id=None,
                cohort_category=category,
            )
            yield query

    def dict(self) -> dict[str, Any]:
        """
        Get a dictionary representation of the cohort definition.
        """
        return {
            "name": self.name,
            "url": self.url,
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
            url=data["url"],
            base_criterion=base_criterion,
            criteria=CriterionCombination.from_dict(data["criteria"]),
        )


class CohortDefinitionCombination(Serializable):
    """
    A cohort definition combination in OMOP as a collection of separate cohort definitions.

    A cohort definition represents an individual recommendation plan (i.e. one part of a single recommendation),
    whereas a cohort definition combination represents the whole recommendation, consisting of one or multiple
    recommendation plans = cohort definitions.
    """

    base_table: Table | None = None

    def __init__(
        self,
        cohort_definitions: list[CohortDefinition],
        base_criterion: Criterion,
        name: str,
        title: str,
        url: str,
        version: str,
        description: str,
        cohort_definition_id: int | None = None,
    ) -> None:
        self._cohort_definitions: list[CohortDefinition] = cohort_definitions
        self._base_criterion: Criterion = base_criterion
        self._name: str = name
        self._title: str = title
        self._url: str = url
        self._version: str = version
        self._description: str = description
        # The id is used in the cohort_definition_id field in the result tables.
        self._id = cohort_definition_id

    @property
    def name(self) -> str:
        """
        Get the name of the recommendation.
        """
        return self._name

    @property
    def title(self) -> str:
        """
        Get the title of the recommendation.
        """
        return self._title

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

    @property
    def description(self) -> str:
        """
        Get the description of the recommendation.
        """
        return self._description

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

    @staticmethod
    def to_table(name: str) -> Table:
        """
        Convert a name to a valid SQL table name.
        """
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        metadata = MetaData()
        return Table(
            name,
            metadata,
            Column("person_id", Integer, primary_key=True),
            Column("valid_date", Date),
        )

    def create_base_table(self) -> Table:
        """
        Create the base table for the cohort definition combination.
        """
        self.base_table = self.to_table(self._base_criterion.name)

        query = self._base_criterion.sql_generate(base_table=self.base_table)
        query = self._base_criterion.sql_insert_into_table(
            query, self.base_table, temporary=True
        )
        assert isinstance(
            query, SelectInto
        ), f"Invalid query - expected SelectInto, got {type(query)}"

        yield query
        yield Index(
            "ix_person_id_valid_date",
            self.base_table.c.person_id,
            self.base_table.c.valid_date,
        )

        # let's also insert the base criterion into the recommendation_result table (for the full list of patients)
        query = add_result_insert(
            self.base_table.select(),
            plan_id=None,
            criterion_id=None,
            cohort_category=CohortCategory.BASE,
        )
        query.description = "Insert base criterion into recommendation_result table"
        yield query

    def process(self) -> Iterator[Query]:
        """
        Process the cohort definition combination into SQL statements.
        """

        yield from self.create_base_table()

        for i, cohort_definition in enumerate(self._cohort_definitions):
            yield from cohort_definition.process(
                base_table=self.base_table,
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
            return (
                select(table.c.person_id, table.c.valid_date)
                .distinct(table.c.person_id, table.c.valid_date)
                .where(
                    and_(
                        table.c.plan_id == cohort_definition.id,
                        table.c.cohort_category == category.name,
                        table.c.criterion_id.is_(None),
                        table.c.recommendation_run_id == bindparam("run_id"),
                    )
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
                query,
                plan_id=None,
                criterion_id=None,
                cohort_category=category,
            )

            yield query

    def cleanup(self) -> Iterator[DropTable]:
        """
        Cleanup the cohort definition combination by removing the temporary base table.
        """
        yield DropTable(self.base_table)

    @staticmethod
    def select_patients(category: CohortCategory) -> Select:
        """
        Select the patients in the given cohort category.
        """

        table = RecommendationResult.__table__

        return select(table.c.person_id).where(
            and_(
                table.c.recommendation_run_id == bindparam("run_id"),
                table.c.plan_id.is_(None),
                table.c.criterion_id.is_(None),
                table.c.cohort_category == category.name,
            )
        )

    def get_criterion(self, criterion_unique_name: str) -> Criterion:
        """
        Retrieve a criterion object by its unique name.
        """

        for cd in self._cohort_definitions:
            criterion = cd.get_criterion(criterion_unique_name)
            if criterion is not None:
                return criterion

        raise ValueError(f"Could not find criterion '{criterion_unique_name}'")

    def criteria(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list
        """
        return list(
            itertools.chain(*[cd.criteria() for cd in self._cohort_definitions])
        )

    def dict(self) -> dict:
        """
        Get the combination as a dictionary.
        """
        return {
            "id": self._id,
            "cohort_definitions": [c.dict() for c in self._cohort_definitions],
            "base_criterion": self._base_criterion.dict(),
            "recommendation_name": self._name,
            "recommendation_title": self._title,
            "recommendation_url": self._url,
            "recommendation_version": self._version,
            "recommendation_description": self._description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create a combination from a dictionary.
        """
        base_criterion = criterion_factory(**data["base_criterion"])
        assert isinstance(
            base_criterion, Criterion
        ), "Base criterion must be a Criterion"

        return cls(
            cohort_definitions=[
                CohortDefinition.from_dict(c) for c in data["cohort_definitions"]
            ],
            base_criterion=base_criterion,
            name=data["recommendation_name"],
            title=data["recommendation_title"],
            url=data["recommendation_url"],
            version=data["recommendation_version"],
            description=data["recommendation_description"],
            cohort_definition_id=data["id"] if "id" in data else None,
        )
