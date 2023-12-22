import logging
from itertools import chain
from typing import Iterator, Type

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination


class ExecutionMap:
    """
    ExecutionMap generates an SQL execution strategy for an Recommendation represented as a collection of criteria.

    The execution map is a tree of criteria that can be executed in a sequential manner. The execution map is generated by
    converting the criterion combination into a negation normal form (NNF) and then flattening the tree into a sequential
    execution map.

    The advantage of the negation normal form (NNF)-based representation of the criterion combination
    is that in the NNF, the negation operator is only applied to single criteria, and not to combinations of criteria.
    This allows us to push the negation operator into the criteria themselves, which simplifies the execution strategy.
    """

    _expr: logic.Expr

    def __init__(
        self, comb: CriterionCombination, base_criterion: Criterion, params: dict | None
    ) -> None:
        """
        Initialize the execution map with a criterion combination.

        :param comb: The criterion combination.
        :param base_criterion: The base criterion which is executed first and used to limit the number of patients
            that are considered for the execution of the other criteria.
        """
        self._expr = self._to_expression(comb, params)
        self._base_criterion = base_criterion

        logging.info(f"Expression: {self._expr}")

    @classmethod
    def from_expression(
        cls, expr: logic.Expr, base_criterion: Criterion
    ) -> "ExecutionMap":
        """
        Create an execution map from a logic expression.

        :param expr: The logic expression.
        :param base_criterion: The base criterion which is executed first and used to limit the number of patients
            that are considered for the execution of the other criteria.
        :return: The execution map.
        """
        instance = cls.__new__(cls)
        instance._expr = expr
        instance._base_criterion = base_criterion

        logging.info(f"Expression: {instance._expr}")

        return instance

    @property
    def expression(self) -> logic.Expr:
        """
        Get the expression of the execution map.
        """
        return self._expr

    @property
    def category(self) -> CohortCategory:
        """
        Get the category of the execution map.
        """
        return self.get_combined_category(self)

    @staticmethod
    def get_combined_category(*emaps: "ExecutionMap") -> CohortCategory:
        """
        Get the combined category of multiple execution maps.

        The combined category is the category of the criterion combination that is created by combining
        the criteria represented by the execution maps.

        BASE is returned only if all execution maps have the category BASE.
        POPULATION is returned if all execution maps have the category POPULATION or BASE.
        INTERVENTION is returned if all execution maps have the category INTERVENTION or BASE.
        POPULATION_INTERVENTION is returned otherwise.

        :param emaps: The execution maps.
        :return: The combined category.
        """
        assert all(
            isinstance(arg, ExecutionMap) for arg in emaps
        ), "all args must be instance of ExecutionMap"

        criteria = list(chain.from_iterable(arg.flatten() for arg in emaps))

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

    def combine(
        self, other: "ExecutionMap", operator: Type[logic.BooleanFunction]
    ) -> "ExecutionMap":
        """
        Combine two execution maps with an operator.
        """
        assert isinstance(other, ExecutionMap), "other must be instance of ExecutionMap"
        assert (
            self._base_criterion == other._base_criterion
        ), "base criteria must be equal"

        category = self.get_combined_category(self, other)

        return ExecutionMap.from_expression(
            operator(self._expr, other._expr, category=category),
            self._base_criterion,
        )

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the execution map.
        """
        self._expr.params = params

    @property
    def params(self) -> dict:
        """
        Get the parameters of the execution map.
        """
        return self._expr.params

    def __and__(self, other: "ExecutionMap") -> "ExecutionMap":
        """
        Combine two execution maps with an AND operator.
        """
        return self.combine(other, logic.And)

    def __or__(self, other: "ExecutionMap") -> "ExecutionMap":
        """
        Combine two execution maps with an OR operator.
        """
        return self.combine(other, logic.Or)

    def __invert__(self) -> "ExecutionMap":
        """
        Invert the execution map.
        """
        return ExecutionMap.from_expression(
            logic.Not(self._expr, category=self.category, params=self._expr.params),
            self._base_criterion,
        )

    def __rshift__(self, other: "ExecutionMap") -> "ExecutionMap":
        """
        Combine two execution maps with an AND operator.
        """
        return self.combine(other, logic.LeftDependentToggle)

    @classmethod
    def combine_from(
        cls, *args: "ExecutionMap", operator: Type[logic.BooleanFunction]
    ) -> "ExecutionMap":
        """
        Combine multiple execution maps with an operator.

        :param args: The execution maps.
        :param operator: The operator.
        :return: The combined execution map.
        """
        assert all(
            isinstance(arg, ExecutionMap) for arg in args
        ), "all args must be instance of ExecutionMap"

        # assert that all arg's base criteria are equal
        assert (
            len(set(arg._base_criterion for arg in args)) == 1
        ), "base criteria must be equal"

        category = cls.get_combined_category(*args)

        return cls.from_expression(
            operator(*[arg._expr for arg in args], category=category),
            args[0]._base_criterion,
        )

    @staticmethod
    def _to_expression(comb: CriterionCombination, params: dict | None) -> logic.Expr:
        """
        Convert the CriterionCombination into an expression of And, Not, Or objects (and possibly more operators).

        :param comb: The criterion combination.
        :param params: The parameters.
        :return: The expression.
        """
        # todo: update docstring

        def conjunction_from_combination(
            comb: CriterionCombination,
        ) -> Type[logic.BooleanFunction]:
            """
            Convert the criterion's operator into a logical conjunction (And or Or)
            """
            if comb.raw_name == "root":
                # todo: this is a hack to make the root node an non-simplifiable Or node - otherwise, using the
                #   sympy.Or,the root node would be simplified to the criterion if there is only one criterion.
                #   The problem is that we need a non-criterion sink node of the intervention and population in order
                #   to store the results to the database without the criterion_id (as the result of the whole
                #   intervention or population of this population/intervention pair).
                return logic.NonSimplifiableOr
            elif comb.operator.operator == CriterionCombination.Operator.AND:
                return logic.And
            elif comb.operator.operator == CriterionCombination.Operator.OR:
                return logic.Or
            else:
                raise NotImplementedError(
                    f'Operator "{str(comb.operator)}" not implemented'
                )

        def _traverse(comb: CriterionCombination) -> logic.Symbol:
            """
            Traverse the criterion combination and creates a collection of logical conjunctions from it.
            """
            conjunction = conjunction_from_combination(comb)
            symbols = []

            for entry in comb:
                if isinstance(entry, CriterionCombination):
                    symbols.append(_traverse(entry))
                else:
                    entry_name = entry.unique_name()
                    s = logic.Symbol(entry_name, criterion=entry, params=params)
                    if entry.exclude:
                        s = logic.Not(s, category=entry.category, params=params)
                    symbols.append(s)

            c = conjunction(*symbols, category=comb.category, params=params)

            if comb.exclude:
                c = logic.Not(c, category=comb.category, params=params)

            return c

        expression = _traverse(comb)

        return expression

    def flatten(self) -> Iterator[Criterion]:
        """
        Flatten the execution map into a list of criteria.
        """
        for atom in self._expr.atoms():
            yield atom.criterion

    def to_graph(self) -> ExecutionGraph:
        """
        Convert the execution map into an execution graph.
        """

        def expression_to_graph(
            expr: logic.Expr,
            graph: ExecutionGraph | None = None,
            parent: logic.Expr | None = None,
        ) -> ExecutionGraph:
            if graph is None:
                graph = ExecutionGraph()

            node_label = expr

            criterion = expr.criterion if expr.is_Atom else None

            graph.add_node(
                node_label,
                criterion=criterion,
                category=expr.category,
                params=expr.params,
                store_result=False,
            )

            if criterion is not None:
                graph.nodes[node_label]["store_result"] = True
                graph.add_edge(base_node_label, node_label)

            if parent is not None:
                graph.add_edge(node_label, parent)

            for arg in expr.args:
                expression_to_graph(arg, graph, node_label)

            return graph

        graph = ExecutionGraph()
        base_node_label = logic.Symbol(
            self._base_criterion.unique_name(), criterion=self._base_criterion
        )
        graph.add_node(
            base_node_label,
            criterion=self._base_criterion,
            category=CohortCategory.BASE,
            store_result=True,
            params={},
        )

        expression_to_graph(self._expr, graph=graph)

        # todo: this is a hack to make the last and penultimate nodes of the population/intervention pair store the result
        #  because the last (=sink) node at the current place this function is called is the combination of all
        #  population/intervention pairs and not the last node of this specific population/intervention pair
        graph.set_sink_nodes_store(hops=1)

        return graph
