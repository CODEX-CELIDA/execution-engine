import copy
import logging
from typing import Iterator, Tuple

import sympy
from sqlalchemy import intersect, union
from sqlalchemy.sql import CompoundSelect

from .omop.criterion.abstract import Criterion
from .omop.criterion.combination import CriterionCombination


class ExecutionMap:
    """
    ExecutionMap generates an SQL execution strategy for an OMOP cohort definition represented as a collection of criteria.

    The execution map is a tree of criteria that can be executed in a sequential manner. The execution map is generated by
    converting the criterion combination into a negation normal form (NNF) and then flattening the tree into a sequential
    execution map.

    The advantage of the negation normal form (NNF)-based representation of the criterion combination (cohort definition) is
    that in the NNF, the negation operator is only applied to single criteria, and not to combinations of criteria. This
    allows us to push the negation operator into the criteria themselves, which simplifies the execution strategy.
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
                raise NotImplementedError(f"Operator {operator} not implemented")

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
                    entry_name = f"{entry}_{hash(entry)}"
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

        for atom in conj.atoms():
            assert conj.count(atom) == 1, f'Duplicate criterion name "{atom}"'

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
                    yield self._hashmap[arg.args[0]].invert_exclude(inplace=True)
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

    def combine(self) -> Tuple[str, list[Criterion]]:
        """
        Generate the combined SQL query for all criteria in the execution map.
        """

        def _traverse(combination: sympy.Symbol) -> CompoundSelect:
            """
            Traverse the execution map and return a combined SQL query for all criteria in the execution map.
            """
            criteria: list[CompoundSelect] = []

            if type(combination) == sympy.And:
                conjunction = intersect
            elif type(combination) == sympy.Or:
                conjunction = union
            else:
                raise ValueError(f"Unknown type {type(combination)}")

            for arg in combination.args:
                if arg.is_Atom:
                    criteria.append(self._hashmap[arg])
                elif arg.is_Not:
                    # exclude flag is already pushed into the criterion (i.e. inverted) in _push_negation_in_criterion()
                    criteria.append(self._hashmap[arg.args[0]])
                else:
                    compound_select = _traverse(arg)
                    criteria.append(compound_select)

            return conjunction(*[c.sql_select(with_alias=False) for c in criteria])

        return _traverse(self._nnf)
