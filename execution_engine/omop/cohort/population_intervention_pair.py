import logging
from typing import Any, Dict

from sqlalchemy import Table
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
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.selectable import CTE

from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.serializable import Serializable
from execution_engine.util.db import add_result_insert
from execution_engine.util.sql import SelectInto


class PopulationInterventionPair(Serializable):
    """
    A population/intervention pair in OMOP as a collection of separate criteria.

    A population/intervention pair represents an individual recommendation plan (i.e. one part of a single recommendation),
    whereas a population/intervention pair combination represents the whole recommendation, consisting of one or multiple
    recommendation plans = population/intervention pairs.
    In turn, a population/intervention pair is a collection of criteria, which can be either a single criterion or a combination
    of criteria (i.e. a criterion combination).
    These single criteria can be either a single criterion (e.g. "has condition X") or a combination of criteria
    (e.g. "has condition X and lab value Y >= Z").
    """

    _id: int | None = None
    _name: str
    _population: CriterionCombination
    _intervention: CriterionCombination
    _execution_map: ExecutionMap | None = None

    def __init__(
        self,
        name: str,
        url: str,
        base_criterion: Criterion,
        population: CriterionCombination | None = None,
        intervention: CriterionCombination | None = None,
    ) -> None:
        self._name = name
        self._url = url
        self._base_criterion = base_criterion
        self._execution_map: ExecutionMap | None = None

        self.set_criteria(CohortCategory.POPULATION, population)
        self.set_criteria(CohortCategory.INTERVENTION, intervention)

    def set_criteria(
        self, category: CohortCategory, criteria: CriterionCombination | None
    ) -> None:
        """
        Set the criteria (either population or intervention) of the population/intervention pair.

        :param category: The category of the criteria.
        :param criteria: The criteria.
        """
        if criteria is None:
            criteria = CriterionCombination(
                name="root",
                exclude=False,
                category=category,
                operator=CriterionCombination.Operator("AND"),
            )
        else:
            assert isinstance(
                criteria, CriterionCombination
            ), f"Invalid criteria - expected CriterionCombination, got {type(criteria)}"
            assert (
                criteria.category == CohortCategory.BASE
            ), f"Invalid criteria - expected category {CohortCategory.BASE}, got {criteria.category}"

        if category == CohortCategory.POPULATION:
            self._population = criteria
        elif category == CohortCategory.INTERVENTION:
            self._intervention = criteria
        else:
            raise ValueError(f"Invalid category {category}")

    @property
    def name(self) -> str:
        """
        Get the name of the population/intervention pair.
        """
        return self._name

    @property
    def url(self) -> str:
        """
        Get the url of the population/intervention pair.
        """
        return self._url

    def execution_map(self) -> dict[CohortCategory, ExecutionMap]:
        """
        Get the execution map for the population/intervention pair.
        """
        # todo can we find a better place to insert this id?
        #     it seems like a mixture of two paradigms, initially
        #     the criteria / tasks etc didn't know about "plans" and so on
        params = {"plan_id": self.id}

        population = ExecutionMap(
            self._population, base_criterion=self._base_criterion, params=params
        )
        intervention = ExecutionMap(
            self._intervention, base_criterion=self._base_criterion, params=params
        )
        return {
            CohortCategory.POPULATION: population,
            CohortCategory.POPULATION: intervention,
        }

    def add_population(self, criterion: Criterion | CriterionCombination) -> None:
        """
        Add a criterion to the population of the population/intervention pair.
        """
        self._population.add(criterion)

    def add_intervention(self, criterion: Criterion | CriterionCombination) -> None:
        """
        Add a criterion to the intervention of the population/intervention pair.
        """
        self._intervention.add(criterion)

    def get_criterion(self, criterion_unique_name: str) -> Criterion | None:
        """
        Retrieve a criterion by its unique name.
        """

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

        return _traverse(self._criteria)"""
        raise NotImplementedError()

    def criteria(self) -> CriterionCombination:
        """
        Get the criteria of the population/intervention pair.
        """
        """return self._criteria"""
        raise NotImplementedError()

    def flatten(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list (i.e. no nested criterion combinations).

        Includes the base criterion, population and intervention.

        :return: A list of individual criteria.
        """

        def _traverse(comb: CriterionCombination) -> list[Criterion]:
            criteria = []
            for element in comb:
                if isinstance(element, CriterionCombination):
                    criteria += _traverse(element)
                else:
                    criteria.append(element)
            return criteria

        return (
            [self._base_criterion]
            + _traverse(self._population)
            + _traverse(self._intervention)
        )

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

    def process(self, base_table: Table) -> None:
        """
        Process the population/intervention pair into SQL statements.
        """
        raise NotImplementedError()
        # self._execution_map = self.execution_map()
        #
        # i: int
        # criterion: Criterion
        #
        # for i, criterion in enumerate(self._execution_map.flatten()):
        #     logging.info(f"Processing {criterion.description()}")
        #
        #     query = criterion.DEPsql_generate(base_table=base_table)
        #     self._assert_base_table_in_select(query, base_table.name)
        #
        #     query = add_result_insert(
        #         query,
        #         plan_id=self.id,
        #         criterion_id=criterion.id,
        #         cohort_category=criterion.category,
        #     )
        #
        #     yield query
        #
        # yield from self.combine()

    def get_criterion_combination_name(self, criterion: Criterion) -> str:
        """
        Get the name of the criterion combination for the given criterion.
        """

        """def _traverse(comb: CriterionCombination) -> str:
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

        raise ValueError(f"Criterion {criterion.name} not found in population/intervention pair")"""
        raise NotImplementedError()

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
        Get a dictionary representation of the population/intervention pair.
        """
        return {
            "name": self.name,
            "url": self.url,
            "base_criterion": self._base_criterion.dict(),
            "population": self._population.dict(),
            "intervention": self._intervention.dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PopulationInterventionPair":
        """
        Create a population/intervention pair from a dictionary.
        """

        base_criterion = criterion_factory(**data["base_criterion"])
        assert isinstance(
            base_criterion, Criterion
        ), "Base criterion must be a Criterion"

        return cls(
            name=data["name"],
            url=data["url"],
            base_criterion=base_criterion,
            population=CriterionCombination.from_dict(data["population"]),
            intervention=CriterionCombination.from_dict(data["intervention"]),
        )
