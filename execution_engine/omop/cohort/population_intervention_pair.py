from typing import Any, Dict, cast

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

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.serializable import Serializable
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

    _name: str
    _population: CriterionCombination
    _intervention: CriterionCombination

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

        self.set_criteria(CohortCategory.POPULATION, population)
        self.set_criteria(CohortCategory.INTERVENTION, intervention)

    def __repr__(self) -> str:
        """
        Get the string representation of the population/intervention pair.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"  name={self._name},\n"
            f"  url={self._url},\n"
            f"  base_criterion={repr(self._base_criterion)},\n"
            f"  population={self._population._repr_pretty(level=1).strip()},\n"
            f"  intervention={self._intervention._repr_pretty(level=1).strip()}\n"
            f")"
        )

    def set_criteria(
        self, category: CohortCategory, criteria: CriterionCombination | None
    ) -> None:
        """
        Set the criteria (either population or intervention) of the population/intervention pair.

        :param category: The category of the criteria.
        :param criteria: The criteria.
        """

        root_combination = LogicalCriterionCombination(
            operator=LogicalCriterionCombination.Operator("AND"),
            root_combination=True,
        )
        if criteria is not None:
            root_combination.add(criteria)

        if category == CohortCategory.POPULATION:
            self._population = root_combination
        elif category == CohortCategory.INTERVENTION:
            self._intervention = root_combination
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

    @classmethod
    def filter_symbols(cls, node: logic.Expr, filter_: logic.Expr) -> logic.Expr:
        """
        Filter (=AND-combine) all symbols by the applied filter function

        Used to filter all intervention criteria (symbols) by the population output in order to exclude
        all intervention events outside the population intervals, which may otherwise interfere with corrected
        determination of temporal combination, i.e. the presence of an intervention event during some time window.
        """

        if isinstance(node, logic.Symbol):
            return logic.LeftDependentToggle(
                left=filter_, right=node, category=CohortCategory.INTERVENTION
            )

        if hasattr(node, "args") and isinstance(node.args, tuple):
            converted_args = [cls.filter_symbols(a, filter_) for a in node.args]

            if any(a is not b for a, b in zip(node.args, converted_args)):
                node.args = tuple(converted_args)

        return node

    def execution_graph(self) -> ExecutionGraph:
        """
        Get the execution graph for the population/intervention pair.
        """

        p = ExecutionGraph.combination_to_expression(
            self._population, category=CohortCategory.POPULATION
        )
        i = ExecutionGraph.combination_to_expression(
            self._intervention, category=CohortCategory.INTERVENTION
        )

        # filter all intervention criteria by the output of the population - this is performed to filter out
        # intervention events that outside of the population intervals (i.e. the time windows during which
        # patients are part of the population) as otherwise events outside of the population time may be picked up
        # by Temporal criteria that determine the presence of some event or condition during a specific time window.
        i = self.filter_symbols(i, filter_=p)

        pi = logic.LeftDependentToggle(
            p,
            i,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        pi_graph = ExecutionGraph.from_expression(pi, self._base_criterion)

        if self._id is None:
            raise ValueError("Population/intervention pair ID not set")

        # todo: should we supply self instead of self._id?
        pi_graph.set_sink_nodes_store(bind_params=dict(pi_pair_id=self._id))

        return pi_graph

    def set_population(self, combination: CriterionCombination) -> None:
        """
        Set the population criteria.
        """
        combination.set_root()
        self._population = combination

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

        Joining the base table ensures that always just a subset of patients is selected,
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

    def dict(self) -> dict[str, Any]:
        """
        Get a dictionary representation of the population/intervention pair.
        """
        base_criterion = self._base_criterion
        population = self._population
        intervention = self._intervention
        return {
            "name": self.name,
            "url": self.url,
            "base_criterion": {
                "class_name": base_criterion.__class__.__name__,
                "data": base_criterion.dict(),
            },
            "population": {
                "class_name": population.__class__.__name__,
                "data": population.dict(),
            },
            "intervention": {
                "class_name": intervention.__class__.__name__,
                "data": intervention.dict(),
            },
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
        population = cast(CriterionCombination, criterion_factory(**data["population"]))
        intervention = cast(
            CriterionCombination, criterion_factory(**data["intervention"])
        )
        object = cls(
            name=data["name"],
            url=data["url"],
            base_criterion=base_criterion,
        )
        # The constructor initializes the population and intervention
        # slots in a particular way, but we want to use whatever we
        # have deserialized instead. This is a bit inefficient because
        # we discard the values that were assigned to the two slots in
        # the constructor.
        object._population = population
        object._intervention = intervention
        return object
