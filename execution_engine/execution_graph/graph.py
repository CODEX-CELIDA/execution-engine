from typing import Any, cast

import networkx as nx

import execution_engine.util.logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.omop.criterion.abstract import Criterion


class ExecutionGraph(nx.DiGraph):
    """
    A directed graph that represents the execution of a cohort query.
    """

    def __add__(self, other: "ExecutionGraph") -> "ExecutionGraph":
        """
        Combine two graphs into one.
        """
        return nx.compose(self, other)

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
        cls, expr: logic.Expr, base_criterion: Criterion, category: CohortCategory
    ) -> "ExecutionGraph":
        """
        Create a graph from a cohort query expression.
        """

        from execution_engine.omop.cohort import PopulationInterventionPairExpr

        expr_hash = hash(expr)

        graph = cls()
        base_node = base_criterion

        graph.add_node(
            base_node,
            category=CohortCategory.BASE,
            store_result=True,
        )

        def traverse(
            expr: logic.Expr,
            parent: logic.Expr | None = None,
            category: CohortCategory = category,
        ) -> None:

            graph.add_node(expr, category=category, store_result=False)

            if parent is not None:
                assert expr in graph.nodes
                assert parent in graph.nodes
                graph.add_edge(expr, parent)

            if isinstance(expr, PopulationInterventionPairExpr):
                # special case for PopulationInterventionPairExpr:
                # we need explicitly set the category of the population and intervention nodes

                p, i = expr.left, expr.right

                traverse(i, parent=expr, category=CohortCategory.INTERVENTION)
                traverse(p, parent=expr, category=CohortCategory.POPULATION)

                # create a subgraph for the pair in order to determine the sink nodes (i.e. the nodes that have no
                # outgoing edges) for POPULATION and POPULATION_INTERVENTION and mark them for storing their result
                # in the database
                subgraph = cast(
                    ExecutionGraph, graph.subgraph(nx.ancestors(graph, expr) | {expr})
                )
                subgraph.set_sink_nodes_store(bind_params=dict(pi_pair_id=expr.id))

            elif expr == base_node:
                # don't need to do anything - only non-base criteria are connected to the base criterion,
                # otherwise we get a cyclic graph
                pass
            elif expr.is_Atom:
                assert expr in graph.nodes, "Node not found in graph"
                graph.nodes[expr]["store_result"] = True
                graph.add_edge(base_node, expr)
            else:
                for child in expr.args:
                    traverse(child, parent=expr, category=category)

        traverse(expr, category=category)

        if hash(expr) != expr_hash:
            raise ValueError("Expression has been modified during traversal")

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

    def sink_nodes(self, category: CohortCategory | None = None) -> list[logic.Expr]:
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

        return sink_nodes

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
                        node.__class__.__name__
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
                    "is_atom": node.is_Atom,
                    "bind_params": self.nodes[node]["bind_params"],
                }
            }

            if isinstance(node, logic.Symbol):

                assert isinstance(
                    node, Criterion
                ), f"Expected Criterion, got {type(node)}"

                node_data["data"]["criterion_id"] = node.id

                def criterion_attr(attr: str) -> str | None:
                    if hasattr(node, attr) and getattr(node, attr) is not None:
                        return str(getattr(node, attr))
                    return None

                try:
                    if node.concept is not None:
                        node_data["data"].update(
                            {
                                "concept": (
                                    node.concept.model_dump()
                                    if node.concept is not None
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
                    node
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
                label = node.description()
            else:
                label = node.__class__.__name__
                symbols = {
                    "And": "&",
                    "Or": "|",
                    "Not": "~",
                    "LeftDependentToggle": "=>",
                    "NonSimplifiableOr": "!|",
                    "NonSimplifiableAnd": "!&",
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
                if self.nodes[predecessor]["category"] == self.nodes[expr]["category"]:
                    set_predecessors_store(predecessor, graph, hops_remaining - 1)

        if desired_category is not None:
            categories = [desired_category]
        else:
            categories = [
                CohortCategory.POPULATION,
                # CohortCategory.INTERVENTION, # we don't store Intervention sink nodes anymore
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
