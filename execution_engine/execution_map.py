import copy
import logging
from typing import Iterator, Tuple

import sympy
from sqlalchemy import CTE, bindparam, intersect, select, union
from sqlalchemy.sql import CompoundSelect, Select

from .constants import CohortCategory
from .omop.criterion.abstract import Criterion
from .omop.criterion.combination import CriterionCombination
from .omop.db.result import RecommendationResult


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

    _nnf: sympy.Expr
    _hashmap: dict[str, Criterion]

    def __init__(self, comb: CriterionCombination) -> None:
        self._nnf, self._hashmap = self._to_nnf(comb)
        self._push_negation_in_criterion()

        logging.info(f"NNF: {self._nnf}")

    @staticmethod
    def _to_nnf(comb: CriterionCombination) -> Tuple[sympy.Expr, dict]:
        """
        Convert the criterion combination into a negation normal form (NNF) and return the NNF expression and a hashmap of
        criteria by their name used in the NNF. This is a workaround because we are using sympy for the NNF conversion
        and sympy seems not to allow adding custom attributes to the expression tree.
        """

        def conjunction_from_operator(
            operator: CriterionCombination.Operator,
        ) -> sympy.Symbol:
            """
            Convert the criterion's operator into a sympy conjunction (And or Or)
            """
            if operator.operator == CriterionCombination.Operator.AND:
                return sympy.And
            elif operator.operator == CriterionCombination.Operator.OR:
                return sympy.Or
            else:
                raise NotImplementedError(f'Operator "{str(operator)}" not implemented')

        def _traverse(
            comb: CriterionCombination,
            hashmap: dict[sympy.Expr, Criterion] | None = None,
        ) -> sympy.Symbol:
            """
            Traverse the criterion combination and creates a collection of sympy conjunctions from it.
            """
            if hashmap is None:
                hashmap = {}

            conjunction = conjunction_from_operator(comb.operator)
            symbols = []

            for entry in comb:
                if isinstance(entry, CriterionCombination):
                    symbols.append(_traverse(entry, hashmap))
                else:
                    entry_name = entry.unique_name()  # f"{entry}_{hash(entry)}"
                    s = sympy.Symbol(entry_name)
                    hashmap[s] = entry
                    if entry.exclude:
                        s = sympy.Not(s)
                    symbols.append(s)

            c = conjunction(*symbols)

            if comb.exclude:
                c = sympy.Not(c)

            return c

        hashmap: dict[sympy.Expr, Criterion] = {}
        conj = _traverse(comb, hashmap)

        # not required anymore: we can use duplicated criteria
        # TODO: however, they should also be executed just once BUT BEWARE: Negations are pushed into the objects!
        #       it is essential to make both negative and positive inclusions of the same criterion possible
        # for atom in conj.atoms():
        #    assert conj.count(atom) == 1, f'Duplicate criterion name "{atom}"'

        return conj.to_nnf(), hashmap

    def _push_negation_in_criterion(self) -> None:
        """
        Push the negation operator into the criteria themselves. This is done by inverting the exclude flag of the
        criteria.

        Note that the hashmap is cloned before, because we do not want to modify the original criteria objects.
        As we are pushing negations from the criterion combination into the criteria, if we would not clone the original
        criteria objects, multiple calls to this function would lead to (incorrect) multiple switchings of the exclude flag.
        """

        self._hashmap = copy.deepcopy(self._hashmap)

        def _flat_traverse(args: tuple[sympy.Expr]) -> None:
            for arg in args:
                if arg.is_Not:
                    self._hashmap[arg.args[0]].exclude = True
                elif arg.is_Atom:
                    self._hashmap[arg].exclude = False
                elif not arg.is_Atom:
                    _flat_traverse(arg.args)

        _flat_traverse(self._nnf.args)

    def nnf(self) -> sympy.Expr:
        """
        Return the negation normal form (NNF) of the criterion combination.
        """
        return self._nnf

    def sequential(self) -> Iterator[Criterion]:
        """
        Traverse the execution map and return a list of criteria that can be executed sequentially.
        """

        def _flat_traverse(args: tuple[sympy.Expr]) -> Iterator[Criterion]:
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

        yield from _flat_traverse(self._nnf.args)

    def combine(self, cohort_category: CohortCategory) -> CompoundSelect:
        """
        Generate the combined SQL query for all criteria in the execution map.
        """

        def fixed_date_range() -> CTE:
            table = RecommendationResult.__table__

            query = (
                select(table.c.person_id, table.c.valid_date)
                .select_from(table)
                .filter(table.c.recommendation_run_id == bindparam("run_id"))
                .filter(table.c.cohort_category == CohortCategory.BASE)
            )

            return query.cte("fixed_date_range")

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

        def _traverse(combination: sympy.Symbol) -> CompoundSelect:
            """
            Traverse the execution map and return a combined SQL query for all criteria in the execution map.
            """
            criteria: list[Criterion | CompoundSelect] = []

            if type(combination) == sympy.And:
                conjunction = intersect
            elif type(combination) == sympy.Or:
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
        query = _traverse(self._nnf)
        query.description = f"Combination of criteria({cohort_category.value})"

        return query
