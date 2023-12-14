import logging
from typing import Iterator

import sympy
from sqlalchemy import (
    CTE,
    Date,
    DateTime,
    bindparam,
    distinct,
    func,
    intersect,
    select,
    union,
)
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql import CompoundSelect, Select
from sqlalchemy.sql.functions import concat

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.db.celida.tables import RecommendationResult
from execution_engine.task.creator import TaskCreator
from execution_engine.task.task import Task


class ExecutionMap:
    """
    ExecutionMap generates an SQL execution strategy for an OMOP cohort definition represented as a collection of criteria.

    The execution map is a tree of criteria that can be executed in a sequential manner. The execution map is generated by
    converting the criterion combination into a negation normal form (NNF) and then flattening the tree into a sequential
    execution map.

    The advantage of the negation normal form (NNF)-based representation of the criterion combination (cohort definition)
    is that in the NNF, the negation operator is only applied to single criteria, and not to combinations of criteria.
    This allows us to push the negation operator into the criteria themselves, which simplifies the execution strategy.
    """

    _expr: logic.Expr
    _hashmap: dict[str, Criterion]

    def __init__(
        self, comb: CriterionCombination, base_criterion: Criterion, params: dict | None
    ) -> None:
        """
        Initialize the execution map with a criterion combination.

        :param comb: The criterion combination.
        :param base_criterion: The base criterion which is executed first and used to limit the number of patients
            that are considered for the execution of the other criteria.
        """
        self._criteria = comb
        self._expr = self._to_expression(self._criteria)
        self._base_criterion = base_criterion
        self._params = params
        self._root_task = self._get_execution_map()

        # self._push_negation_in_criterion()

        logging.info(f"NNF: {self._expr}")

    def root_task(self) -> Task:
        """
        Return the root task of the execution map.
        """
        return self._root_task

    @staticmethod
    def get_combined_category(*args: "ExecutionMap") -> CohortCategory:
        """
        Get the combined category of multiple execution maps.

        The combined category is the category of the cohort definition that is created by combining
        the cohort definitions represented by the execution maps.

        BASE is returned only if all execution maps have the category BASE.
        POPULATION is returned if all execution maps have the category POPULATION or BASE.
        INTERVENTION is returned if all execution maps have the category INTERVENTION or BASE.
        POPULATION_INTERVENTION is returned otherwise.

        :param args: The execution maps.
        :return: The combined category.
        """
        assert all(
            isinstance(arg, ExecutionMap) for arg in args
        ), "all args must be instance of ExecutionMap"

        criteria = [arg._criteria for arg in args]

        if all(c.category == CohortCategory.BASE for c in criteria):
            category = CohortCategory.BASE
        elif all(
            c.category == CohortCategory.POPULATION or c.category == CohortCategory.BASE
            for c in criteria
        ):
            category = CohortCategory.POPULATION
        elif all(
            c.category == CohortCategory.INTERVENTION
            or c.category == CohortCategory.BASE
            for c in criteria
        ):
            category = CohortCategory.INTERVENTION
        else:
            category = CohortCategory.POPULATION_INTERVENTION

        return category

    def __and__(self, other: "ExecutionMap") -> "ExecutionMap":
        """
        Combine two execution maps with an AND operator.
        """
        assert isinstance(other, ExecutionMap), "other must be instance of ExecutionMap"
        assert (
            self._base_criterion == other._base_criterion
        ), "base criteria must be equal"

        assert (
            self._params is None and other._params is None
        ) or self._params == other._params, "params must be equal"

        criteria = [self._criteria, other._criteria]
        category = self.get_combined_category(self, other)

        return ExecutionMap(
            CriterionCombination(
                "AND",  # todo: proper name
                exclude=False,
                operator=CriterionCombination.Operator(
                    CriterionCombination.Operator.AND
                ),
                category=category,
                criteria=criteria,
            ),
            self._base_criterion,
            self._params,
        )

    def __or__(self, other: "ExecutionMap") -> "ExecutionMap":
        """
        Combine two execution maps with an OR operator.
        """
        assert isinstance(other, ExecutionMap), "other must be instance of ExecutionMap"
        assert (
            self._base_criterion == other._base_criterion
        ), "base criteria must be equal"

        assert (
            self._params is None and other._params is None
        ) or self._params == other._params, "params must be equal"

        criteria = [self._criteria, other._criteria]
        category = self.get_combined_category(self, other)

        return ExecutionMap(
            CriterionCombination(
                "OR",  # todo: proper name
                exclude=False,
                operator=CriterionCombination.Operator(
                    CriterionCombination.Operator.OR
                ),
                category=category,
                criteria=criteria,
            ),
            self._base_criterion,
            self._params,
        )

    def __invert__(self) -> "ExecutionMap":
        """
        Invert the execution map.
        """
        return ExecutionMap(self._criteria.invert(), self._base_criterion, self._params)

    @classmethod
    def union(cls, *args: "ExecutionMap") -> "ExecutionMap":
        """
        Combine multiple execution maps with an OR operator.
        """
        assert all(
            isinstance(arg, ExecutionMap) for arg in args
        ), "all args must be instance of ExecutionMap"

        # assert that all arg's base criteria are equal
        assert (
            len(set(arg._base_criterion for arg in args)) == 1
        ), "base criteria must be equal"

        assert all(arg._params is None for arg in args) or (
            len(set(arg._params for arg in args)) == 1
        ), "params must be equal"

        criteria = [arg._criteria for arg in args]
        category = cls.get_combined_category(*args)

        return cls(
            CriterionCombination(
                "OR",
                exclude=False,
                operator=CriterionCombination.Operator(
                    CriterionCombination.Operator.OR
                ),
                category=category,
                criteria=criteria,
            ),
            args[0]._base_criterion,
            args[0]._params,
        )

    def _get_execution_map(self) -> Task:
        def count_usage(expr: logic.Expr, usage_count: dict[logic.Expr, int]) -> None:
            usage_count[expr] = usage_count.get(expr, 0) + 1
            if not expr.is_Atom:
                for arg in expr.args:
                    count_usage(arg, usage_count)

        usage_counts: dict[logic.Expr, int] = {}
        count_usage(self._expr, usage_counts)

        tc = TaskCreator(base_criterion=self._base_criterion, params=self._params)

        return tc.create_tasks_and_dependencies(self._expr)

    @staticmethod
    def _to_expression(comb: CriterionCombination) -> logic.Expr:
        """
        Convert the criterion combination into a logic expression and a hashmap of
        criteria by their name used in the NNF. This is a workaround because we are using sympy for the NNF conversion
        and sympy seems not to allow adding custom attributes to the expression tree.
        """
        # todo: update docstring

        def conjunction_from_operator(
            operator: CriterionCombination.Operator,
        ) -> sympy.Expr:
            """
            Convert the criterion's operator into a sympy conjunction (And or Or)
            """
            if operator.operator == CriterionCombination.Operator.AND:
                return logic.And
            elif operator.operator == CriterionCombination.Operator.OR:
                return logic.Or
            else:
                raise NotImplementedError(f'Operator "{str(operator)}" not implemented')

        def _traverse(comb: CriterionCombination) -> logic.Symbol:
            """
            Traverse the criterion combination and creates a collection of sympy conjunctions from it.
            """
            conjunction = conjunction_from_operator(comb.operator)
            symbols = []

            for entry in comb:
                if isinstance(entry, CriterionCombination):
                    symbols.append(_traverse(entry))
                else:
                    entry_name = entry.unique_name()
                    s = logic.Symbol(entry_name, criterion=entry)
                    if entry.exclude:
                        s = logic.Not(s, category=entry.category)
                    symbols.append(s)

            c = conjunction(*symbols, category=comb.category)

            if comb.exclude:
                c = logic.Not(c, category=comb.category)

            return c

        conj = _traverse(comb)

        return conj

    def flatten(self) -> Iterator[Criterion]:
        """
        Traverse the execution map and return a list of criteria that can be executed sequentially.
        """

        def _flat_traverse(args: tuple[logic.Expr]) -> Iterator[Criterion]:
            """
            Traverse the execution map and return a list of criteria that can be executed sequentially.
            """
            for arg in args:
                if arg.is_Atom:
                    yield self._hashmap[arg]
                elif arg.is_Not:
                    # exclude flag is already pushed into the criterion (i.e. inverted) in _push_negation_in_criterion()
                    yield self._hashmap[arg.args[0]]
                else:
                    yield from _flat_traverse(arg.args)

        yield from _flat_traverse(self._expr.args)

    def combine(self, cohort_category: CohortCategory) -> CompoundSelect:
        """
        Generate the combined SQL query for all criteria in the execution map.
        """

        def select_base_criterion() -> CTE:
            table = RecommendationResult.__table__

            query = (
                select(table.c.person_id, table.c.valid_date)
                .select_from(table)
                .filter(table.c.recommendation_run_id == bindparam("run_id"))
                .filter(table.c.cohort_category == CohortCategory.BASE)
            )

            return query

        def fixed_date_range() -> CTE:
            table = RecommendationResult.__table__

            distinct_persons = (
                select(distinct(table.c.person_id).label("person_id"))
                .select_from(table)
                .filter(table.c.recommendation_run_id == bindparam("run_id"))
                .filter(table.c.cohort_category == CohortCategory.BASE)
            ).cte("distinct_persons")

            fixed_date_range = (
                select(
                    distinct_persons.c.person_id,
                    func.generate_series(
                        bindparam("observation_start_datetime", type_=DateTime).cast(
                            Date
                        ),
                        bindparam("observation_end_datetime", type_=DateTime).cast(
                            Date
                        ),
                        func.cast(concat(1, "day"), INTERVAL),
                    )
                    .cast(Date)
                    .label("valid_date"),
                )
                .select_from(distinct_persons)
                .cte("fixed_date_range")
            )

            return fixed_date_range

        def sql_select(criterion: Criterion | CompoundSelect, exclude: bool) -> Select:
            """
            Generate the SQL query for a single criterion.
            """
            if isinstance(criterion, CompoundSelect):
                return criterion
            table = RecommendationResult.__table__

            query = (
                select(table.c.person_id, table.c.valid_date)
                .select_from(table)
                .filter(table.c.recommendation_run_id == bindparam("run_id"))
                .filter(table.c.criterion_id == criterion.id)
            )

            if exclude:
                query = add_exclude(query)

            return query

        def add_exclude(query: str | Select) -> Select:
            """
            Converts a query, which returns a list of dates (one per row) per person_id, to a query that return a list of
            all days (per person_id) that are within the time range given by observation_start_datetime
            and observation_end_datetime but that are not included in the result of the original query.

            I.e. it performs the following set operation:
            set({day | observation_start_datetime <= day <= observation_end_datetime}) - set(days_from_original_query}
            """
            assert isinstance(query, Select | CTE), "query must be instance of Select"

            query = query.alias("person_dates")

            query = (
                select(
                    cte_fixed_date_range.c.person_id, cte_fixed_date_range.c.valid_date
                )
                .select_from(
                    cte_fixed_date_range.outerjoin(
                        query,
                        (cte_fixed_date_range.c.person_id == query.c.person_id)
                        & (cte_fixed_date_range.c.valid_date == query.c.valid_date),
                    )
                )
                .where(query.c.valid_date.is_(None))
            )

            return query

        def _traverse(combination: logic.Symbol) -> CompoundSelect:
            """
            Traverse the execution map and return a combined SQL query for all criteria in the execution map.
            """
            criteria: list[Criterion | CompoundSelect] = []

            if isinstance(combination, logic.And):
                conjunction = intersect
            elif isinstance(combination, logic.Or):
                conjunction = union
            else:
                raise ValueError(f"Unknown type {type(combination)}")

            criterion: Criterion | CompoundSelect

            for arg in combination.args:
                if arg.is_Atom:
                    criterion, exclude = self._hashmap[arg], False
                elif arg.is_Not:
                    # exclude flag is already pushed into the criterion (i.e. inverted) in _push_negation_in_criterion()
                    criterion, exclude = self._hashmap[arg.args[0]], True
                else:
                    criterion, exclude = _traverse(arg), False

                if isinstance(criterion, Criterion):
                    if (
                        cohort_category is not CohortCategory.POPULATION_INTERVENTION
                        and criterion.category != cohort_category
                    ):
                        # this criterion does not belong to the cohort category we are currently processing,
                        # so we skip it
                        continue
                elif isinstance(criterion, CompoundSelect):
                    if len(criterion.selects) == 0:
                        # this depth-first traversal returns a compound select (union or intersect) of the already
                        # processed (deeper) criteria. If all the deeper criteria do not belong to the cohort category
                        # being processed, none of them is added to the compound select and an empty compound select is
                        # returned. In this case, we skip the compound select and continue with the next criterion.
                        continue

                criteria.append([criterion, exclude])

            return conjunction(
                *[sql_select(criterion, exclude) for [criterion, exclude] in criteria]
            )

        cte_fixed_date_range = fixed_date_range()

        query = _traverse(self._expr)
        query = intersect(select_base_criterion(), query)
        query.description = f"Combination of criteria({cohort_category.value})"

        return query
