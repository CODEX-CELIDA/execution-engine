import copy

import networkx as nx

import execution_engine.util.logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.omop.cohort.population_intervention_pair import (
    PopulationInterventionPairExpr,
)
from execution_engine.omop.criterion.abstract import Criterion


class RecommendationGraphBuilder:
    """
    A builder class for constructing ExecutionGraph objects based on
    population/intervention expressions. It provides utility methods to filter
    intervention criteria by population constraints, and then converts the
    filtered expression into an ExecutionGraph ready for execution and storage.
    """

    @classmethod
    def filter_symbols(cls, node: logic.Expr, filter_: logic.Expr) -> logic.Expr:
        """
        Filter (=AND-combine) all symbols by the applied filter function

        Used to filter all intervention criteria (symbols) by the population output in order to exclude
        all intervention events outside the population intervals, which may otherwise interfere with corrected
        determination of temporal combination, i.e. the presence of an intervention event during some time window.

        :param node: The expression node to be filtered.
        :type node: logic.Expr
        :param filter_: The filter expression to AND-combine with symbols in the node.
        :type filter_: logic.Expr
        :return: A new expression in which all symbols are constrained by the filter expression.
        :rtype: logic.Expr
        """

        node = copy.copy(node)

        if isinstance(node, logic.Symbol):
            return logic.LeftDependentToggle(left=filter_, right=node)
        elif isinstance(node, logic.Expr):
            if hasattr(node, "interval_criterion") and node.interval_criterion:
                # we must not wrap the interval_criterion
                interval_criterion = node.interval_criterion
                converted_args = [
                    cls.filter_symbols(a, filter_)
                    for a in node.args
                    if not a == interval_criterion
                ] + [interval_criterion]
            else:
                converted_args = [cls.filter_symbols(a, filter_) for a in node.args]

            if any(a is not b for a, b in zip(node.args, converted_args)):
                node.update_args(*converted_args)

        return node

    @classmethod
    def filter_intervention_criteria_by_population(cls, expr: logic.Expr) -> logic.Expr:
        """
        Filter all intervention criteria in a given expression by the population part of the expression.

        :param expr: The expression that may contain population and intervention parts.
        :type expr: logic.Expr
        :return: A new expression where all intervention symbols are constrained by the population intervals.
        :rtype: logic.Expr
        """

        from execution_engine.omop.cohort import PopulationInterventionPairExpr

        # we might make changes to the expression (e.g. filtering), so we must preserve
        # the original expression from the caller
        expr = copy.deepcopy(expr)

        def traverse(
            expr: logic.Expr,
        ) -> None:
            if isinstance(expr, PopulationInterventionPairExpr):
                p, i = expr.left, expr.right

                # filter all intervention criteria by the output of the population - this is performed to filter out
                # intervention events that outside of the population intervals (i.e. the time windows during which
                # patients are part of the population) as otherwise events outside of the population time may be picked up
                # by Temporal criteria that determine the presence of some event or condition during a specific time window.
                i = cls.filter_symbols(i, filter_=p)

                expr.update_args(p, i)

                traverse(i)
                traverse(p)

            elif not expr.is_Atom:
                for child in expr.args:
                    traverse(child)

        traverse(expr)

        # we need to rehash as the structure has been changed due to insertion of additional nodes
        expr.rehash(recursive=True)

        return expr

    @classmethod
    def build(cls, expr: logic.Expr, base_criterion: Criterion) -> ExecutionGraph:
        """
        Build an ExecutionGraph for a population/intervention expression.

        If the expression is a PopulationInterventionPairExpr, it is wrapped in a
        NonSimplifiableAnd to ensure a top-level result entry is generated in the database.
        Then the expression is filtered and converted into an ExecutionGraph with the
        appropriate sink nodes and edges.

        :param expr: The population/intervention expression to build the graph from.
        :type expr: logic.Expr
        :param base_criterion: The base criterion used to label the execution graph.
        :type base_criterion: Criterion
        :return: The constructed ExecutionGraph for the given expression.
        :rtype: ExecutionGraph
        """
        if isinstance(expr, PopulationInterventionPairExpr):
            expr = logic.NonSimplifiableAnd(expr)

        # Make sure the expr is filtered
        expr_filtered = cls.filter_intervention_criteria_by_population(expr)

        graph = ExecutionGraph.from_expression(
            expr_filtered,
            base_criterion=base_criterion,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        p_sink_nodes = graph.sink_nodes(CohortCategory.POPULATION)
        graph.set_sink_nodes_store(
            bind_params={}, desired_category=CohortCategory.POPULATION_INTERVENTION
        )

        p_combination_node = logic.NonSimplifiableOr(*p_sink_nodes)
        graph.add_node(
            p_combination_node, store_result=True, category=CohortCategory.POPULATION
        )
        graph.add_edges_from((src, p_combination_node) for src in p_sink_nodes)

        if graph.in_degree(base_criterion) != 0:
            raise AssertionError("Base criterion must not have incoming edges")

        if not nx.is_directed_acyclic_graph(graph):
            raise AssertionError("Graph is not acyclic")

        return graph
