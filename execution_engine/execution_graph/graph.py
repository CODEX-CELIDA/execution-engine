from typing import Any, Callable, Type

import networkx as nx

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.combination import CriterionCombination
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
    NonCommutativeLogicalCriterionCombination,
)
from execution_engine.omop.criterion.combination.temporal import (
    FixedWindowTemporalIndicatorCombination,
    PersonalWindowTemporalIndicatorCombination,
    TemporalIndicatorCombination,
)


class ExecutionGraph(nx.DiGraph):
    """
    A directed graph that represents the execution of a cohort query.
    """

    def __add__(self, other: "ExecutionGraph") -> "ExecutionGraph":
        """
        Combine two graphs into one.
        """
        return nx.compose(self, other)

    @classmethod
    def from_criterion_combination(
        cls, combination: LogicalCriterionCombination, base_criterion: Criterion
    ) -> "ExecutionGraph":
        """
        Create a graph from a population and intervention criterion combination.
        """
        return cls.from_expression(
            cls.combination_to_expression(combination), base_criterion
        )

    def is_sink_of_category(
        self, expr: logic.Expr, graph: "ExecutionGraph", category: CohortCategory
    ) -> bool:
        """
        Check if a node is a sink node of the graph of a given category.
        """
        if self.nodes[expr]["category"] != category:
            return False  # Node is not of the desired category

        for _, neighbor in graph.out_edges(expr):
            if self.nodes[neighbor]["category"] == category:
                return False  # Node has an outgoing edge to a node of the same category

        return True

    def is_sink(self, expr: logic.Expr) -> bool:
        """
        Check if a node is a sink node of the graph.
        """
        return self.out_degree(expr) == 0

    @classmethod
    def from_expression(
        cls, expr: logic.Expr, base_criterion: Criterion
    ) -> "ExecutionGraph":
        """
        Create a graph from a cohort query expression.
        """

        def expression_to_graph(
            expr: logic.Expr,
            graph: ExecutionGraph,
            parent: logic.Expr | None = None,
        ) -> ExecutionGraph:
            graph.add_node(expr, category=expr.category, store_result=False)

            if expr.is_Atom:
                graph.nodes[expr]["store_result"] = True
                graph.add_edge(base_node, expr)

            if parent is not None:
                graph.add_edge(expr, parent)

            for child in expr.args:
                expression_to_graph(child, graph, expr)

            return graph

        graph = cls()
        base_node = logic.Symbol(
            criterion=base_criterion,
        )
        graph.add_node(
            base_node,
            category=CohortCategory.BASE,
            store_result=True,
        )

        expression_to_graph(expr, graph=graph)

        return graph

    def add_node(
        self,
        node_for_adding: logic.BaseExpr,
        category: CohortCategory,
        store_result: bool,
        bind_params: dict | None = None,
    ) -> None:
        """
        Add a node to the graph.

        :param node_for_adding: The node to add.
        :param category: The category of the node.
        :param store_result: Whether the result of the node should be stored.
        :param bind_params: The parameters to bind to the sql query.
        """

        super().add_node(
            node_for_adding,
            category=category,
            store_result=store_result,
            bind_params=bind_params if bind_params is not None else {},
        )

    @classmethod
    def combine_from(cls, *graphs: "ExecutionGraph") -> "ExecutionGraph":
        """
        Combine multiple graphs into one.
        """
        combined_graph = cls()
        for graph in graphs:
            combined_graph = nx.compose(combined_graph, graph)

        return combined_graph

    def sink_node(self, category: CohortCategory | None = None) -> logic.Expr:
        """
        Get the sink node of the graph.

        The sink node is the node that has no outgoing edges.
        """

        if category is None:
            sink_nodes = [node for node in self.nodes() if self.out_degree(node) == 0]
        else:
            sink_nodes = [
                node
                for node in self.nodes()
                if self.is_sink_of_category(node, self, category)
            ]

        if len(sink_nodes) != 1:
            raise ValueError(
                f"There must be exactly one sink node, but there are {len(sink_nodes)}"
            )

        return sink_nodes[0]

    def to_cytoscape_dict(self) -> dict:
        """
        Convert the graph to a dictionary that can be used by Cytoscape.js.
        """
        nodes = []
        edges = []

        for node in self.nodes():
            # Ensure all node attributes are serializable

            node_data = {
                "data": {
                    "id": id(node),
                    "label": str(node),
                    "class": (
                        node.criterion.__class__.__name__
                        if isinstance(node, logic.Symbol)
                        else node.__class__.__name__
                    ),
                    "type": (
                        node._repr_join_str
                        if hasattr(node, "_repr_join_str")
                        and node._repr_join_str is not None
                        else node.__class__.__name__
                    ),
                    "category": self.nodes[node][
                        "category"
                    ].value,  # Assuming 'value' is serializable
                    "store_result": str(
                        self.nodes[node]["store_result"]
                    ),  # Convert to string if necessary
                    "is_sink": self.is_sink(node),
                    "bind_params": self.nodes[node]["bind_params"],
                }
            }

            if isinstance(node, logic.Symbol):

                node_data["data"]["criterion_id"] = node.criterion._id

                def criterion_attr(attr: str) -> str | None:
                    if (
                        hasattr(node.criterion, attr)
                        and getattr(node.criterion, attr) is not None
                    ):
                        return str(getattr(node.criterion, attr))
                    return None

                try:
                    if node.criterion.concept is not None:
                        node_data["data"].update(
                            {
                                "concept": (
                                    node.criterion.concept.model_dump()
                                    if node.criterion.concept is not None
                                    else None
                                ),
                                "value": criterion_attr("value"),
                                "timing": criterion_attr("timing"),
                                "dose": criterion_attr("dose"),
                                "route": criterion_attr("route"),
                            }
                        )
                except NotImplementedError:
                    # non-concept criterion, e.g. base criterion
                    pass
            elif isinstance(node, logic.TemporalCount):
                node_data["data"]["start_time"] = node.start_time
                node_data["data"]["end_time"] = node.end_time
                node_data["data"]["interval_type"] = node.interval_type
                node_data["data"]["interval_criterion"] = repr(node.interval_criterion)

            if hasattr(node, "count_min"):
                node_data["data"]["count_min"] = node.count_min
            if hasattr(node, "count_max"):
                node_data["data"]["count_max"] = node.count_max

            if self.nodes[node]["category"] == CohortCategory.BASE:
                node_data["data"]["base_criterion"] = str(
                    node.criterion
                )  # Ensure this is serializable

            nodes.append(node_data)

        for edge in self.edges():
            edges.append(
                {
                    "data": {
                        "source": id(edge[0]),
                        "target": id(edge[1]),
                    }
                }
            )

        return {"nodes": nodes, "edges": edges}

    def plot(self) -> None:
        """
        Plot the graph.
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plotting")

        plt.figure(figsize=(20, 20))
        pos = nx.kamada_kawai_layout(self)

        node_colors = [
            "green" if self.nodes[node]["store_result"] else "red"
            for node in self.nodes()
        ]

        labels = {}
        for node in self.nodes:
            category = {
                CohortCategory.POPULATION: "P",
                CohortCategory.INTERVENTION: "I",
                CohortCategory.POPULATION_INTERVENTION: "PI",
                CohortCategory.BASE: "B",
            }[self.nodes[node]["category"]]

            if node.is_Atom:
                label = node.criterion.description()
            else:
                label = node.__class__.__name__
                symbols = {
                    "And": "&",
                    "Or": "|",
                    "Not": "~",
                    "LeftDependentToggle": "=>",
                    "NonSimplifiableOr": "!|",
                    "NonSimplifiableAnd": "!&",
                    "NoDataPreservingAnd": "NDP-&",
                    "NoDataPreservingOr": "NPD-|",
                    "MinCount": "Min",
                    "MaxCount": "Max",
                    "ExactCount": "Exact",
                }
                label = symbols.get(label, label)

            labels[node] = label + f" [{category}]"

        nx.draw(
            self,
            pos,
            labels=labels,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=2500,
            font_size=10,
        )

        plt.title("Expression Graph")
        plt.show()

    def set_sink_nodes_store(
        self,
        *,
        bind_params: dict,
        hops: int = 0,
        desired_category: CohortCategory | None = None,
    ) -> None:
        """
        Set the store_result flag for all sink nodes of the graph.

        Find all nodes of the desired category that have no outgoing edges to nodes of the same category
        these are the last nodes of POPULATION, INTERVENTION or POPULATION_INTERVENTION and their result should
        be stored.

        :param bind_params: The parameters to bind to the sql query.
        :param hops: The number of hops to go back from the sink nodes.
        :param desired_category: The category of the sink nodes. None means all categories.
        """

        def set_predecessors_store(
            expr: logic.Expr, graph: ExecutionGraph, hops_remaining: int
        ) -> None:
            self.nodes[expr]["store_result"] = True
            self.nodes[expr]["bind_params"] = bind_params

            if hops_remaining <= 0:
                return
            for predecessor in graph.predecessors(expr):
                if predecessor.category == expr.category:
                    set_predecessors_store(predecessor, graph, hops_remaining - 1)

        if desired_category is not None:
            categories = [desired_category]
        else:
            categories = [
                CohortCategory.POPULATION,
                CohortCategory.INTERVENTION,
                CohortCategory.POPULATION_INTERVENTION,
            ]

        for category in categories:
            sink_nodes_of_desired_category = [
                node
                for node in self.nodes()
                if self.is_sink_of_category(node, self, category)
            ]
            assert (
                len(sink_nodes_of_desired_category) <= 1
            ), "There must be zero or one sink nodes of the desired category"

            for sink_node in sink_nodes_of_desired_category:
                set_predecessors_store(sink_node, self, hops)

    @staticmethod
    def combination_to_expression(comb: CriterionCombination) -> logic.Expr:
        """
        Convert the CriterionCombination into an expression of And, Not, Or objects (and possibly more operators).

        :param comb: The criterion combination.
        :return: The expression.
        """

        def conjunction_from_combination(
            comb: CriterionCombination,
        ) -> Type[logic.BooleanFunction] | Callable:
            """
            Convert the criterion's operator into a logical conjunction (Not or And or Or)
            """
            if isinstance(comb, LogicalCriterionCombination):
                if comb.is_root():
                    # This is a hack to make the root node a non-simplifiable And node - otherwise, using the
                    #   logic.And, the root node would be simplified to the criterion if there is only one criterion.
                    #   The problem is that we need a non-criterion sink node of the intervention and population in order
                    #   to store the results to the database without the criterion_id (as the result of the whole
                    #   intervention or population of this population/intervention pair).

                    if (
                        comb.operator.operator
                        != LogicalCriterionCombination.Operator.AND
                    ):
                        raise AssertionError(
                            f"Invalid operator {comb.operator} for root node. Expected AND."
                        )
                    return logic.NonSimplifiableAnd

                    # Handle non-commutative combinations.
                if isinstance(comb, NonCommutativeLogicalCriterionCombination):
                    return logic.ConditionalFilter

                op = comb.operator.operator

                # Mapping of simple logical operators.
                simple_ops = {
                    LogicalCriterionCombination.Operator.NOT: logic.Not,
                    LogicalCriterionCombination.Operator.AND: logic.And,
                    LogicalCriterionCombination.Operator.OR: logic.Or,
                    LogicalCriterionCombination.Operator.ALL_OR_NONE: logic.AllOrNone,
                }
                if op in simple_ops:
                    return simple_ops[op]

                # Mapping of count-based operators.
                count_ops = {
                    LogicalCriterionCombination.Operator.AT_LEAST: logic.MinCount,
                    LogicalCriterionCombination.Operator.AT_MOST: logic.MaxCount,
                    LogicalCriterionCombination.Operator.EXACTLY: logic.ExactCount,
                }
                if op in count_ops:
                    if comb.operator.threshold is None:
                        raise ValueError(
                            f"Threshold must be set for operator {comb.operator.operator}"
                        )
                    return lambda *args, category: count_ops[op](
                        *args, threshold=comb.operator.threshold, category=category
                    )

                raise NotImplementedError(f'Operator "{comb.operator}" not implemented')

            ###################################################################################
            elif isinstance(comb, TemporalIndicatorCombination):

                interval_criterion: logic.BaseExpr | None = None
                start_time = None
                end_time = None
                interval_type = None

                if isinstance(comb, PersonalWindowTemporalIndicatorCombination):

                    if isinstance(comb.interval_criterion, CriterionCombination):
                        interval_criterion = _traverse(comb.interval_criterion)
                    elif isinstance(comb.interval_criterion, Criterion):
                        interval_criterion = logic.Symbol(comb.interval_criterion)
                    else:
                        raise ValueError(
                            f"Invalid interval criterion type: {type(comb.interval_criterion)}"
                        )

                elif isinstance(comb, FixedWindowTemporalIndicatorCombination):
                    start_time = comb.start_time
                    end_time = comb.end_time
                    interval_type = comb.interval_type

                # Ensure a threshold is set.
                if comb.operator.threshold is None:
                    raise ValueError(
                        f"Threshold must be set for operator {comb.operator.operator}"
                    )

                # Map the operator to the corresponding logic function.
                op_map = {
                    TemporalIndicatorCombination.Operator.AT_LEAST: logic.TemporalMinCount,
                    TemporalIndicatorCombination.Operator.AT_MOST: logic.TemporalMaxCount,
                    TemporalIndicatorCombination.Operator.EXACTLY: logic.TemporalExactCount,
                }
                op_func = op_map.get(comb.operator.operator, None)
                if op_func is None:
                    raise NotImplementedError(
                        f'Operator "{str(comb.operator)}" not implemented'
                    )

                return lambda *args, category: op_func(
                    *args,
                    threshold=comb.operator.threshold,
                    category=category,
                    start_time=start_time,
                    end_time=end_time,
                    interval_type=interval_type,
                    interval_criterion=interval_criterion,
                )

            else:
                raise ValueError(f"Invalid combination type: {type(comb)}")

        def _traverse(comb: CriterionCombination) -> logic.Expr:
            """
            Traverse the criterion combination and creates a collection of logical conjunctions from it.
            """
            conjunction = conjunction_from_combination(comb)
            components: list[logic.Expr | logic.Symbol] = []

            for entry in comb:
                if isinstance(entry, CriterionCombination):
                    components.append(_traverse(entry))
                elif isinstance(entry, Criterion):
                    components.append(logic.Symbol(entry))
                else:
                    raise ValueError(f"Invalid entry type: {type(entry)}")

            return conjunction(*components, category=comb.category)

        expression = _traverse(comb)

        return expression

    def __eq__(self, other: Any) -> bool:
        """
        Check if two graphs are equal.
        """
        if not isinstance(other, ExecutionGraph):
            return False

        # Compare nodes and their attributes
        for node in self:
            if node not in other or self.nodes[node] != other.nodes[node]:
                return False

        # Compare edges and their attributes
        for edge in self.edges:
            if edge not in other.edges or self.edges[edge] != other.edges[edge]:
                return False

        # Check if 'other' has extra nodes or edges
        if len(self.nodes) != len(other.nodes) or len(self.edges) != len(other.edges):
            return False

        return True


def merge_graphs(graphs: list[ExecutionGraph]) -> ExecutionGraph:
    """
    Merges multiple directed graphs into a single graph, combining node attributes.

    This function iterates over a list of NetworkX DiGraphs and merges them into a single DiGraph.
    If a node is present in multiple graphs, its attributes are merged. Specifically, for the 'bind_params'
    attribute, the function combines the values into a list if they are different, ensuring that all distinct
    values from the different graphs are retained.

    Parameters:
    graphs (list[nx.DiGraph]): A list of NetworkX DiGraphs to be merged.

    Returns:
    nx.DiGraph: A new directed graph containing all nodes and edges from the input graphs.
                Node attributes are merged where applicable.

    Note:
    - The function assumes that the 'bind_params' attribute in the nodes, if present, is a dictionary.
    - Edge attributes are copied as-is without conflict resolution. If conflict resolution for edges is needed,
      this function should be modified accordingly.

    Example:
    >>> graph1 = nx.DiGraph()
    >>> graph1.add_node(1, bind_params={'param1': 'value1'})
    >>> graph1.add_edge(1, 2)
    >>> graph2 = nx.DiGraph()
    >>> graph2.add_node(1, bind_params={'param1': 'value2'})
    >>> graph2.add_edge(1, 2)
    >>> merged_graph = merge_graphs([graph1, graph2])
    >>> merged_graph.nodes[1]['bind_params']['param1']
    ['value1', 'value2']
    """
    merged_graph = ExecutionGraph()

    for graph in graphs:
        for node, attrs in graph.nodes(data=True):
            if node in merged_graph:
                merged_attr = merged_graph.nodes[node]

                for key in merged_attr["bind_params"]:
                    if key in attrs["bind_params"]:
                        if attrs["bind_params"][key] == merged_attr["bind_params"][key]:
                            continue

                        if not isinstance(merged_attr["bind_params"][key], list):
                            merged_attr["bind_params"][key] = [
                                merged_attr["bind_params"][key],
                                attrs["bind_params"][key],
                            ]
                        else:
                            merged_attr["bind_params"][key].append(
                                attrs["bind_params"][key]
                            )
            else:
                # Add new node with attributes
                merged_graph.add_node(node, **attrs)

        for edge in graph.edges(data=True):
            merged_graph.add_edge(*edge[:2], **edge[2])

    return merged_graph
