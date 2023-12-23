import networkx as nx
from matplotlib import pyplot as plt

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory


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
    def combine_from(cls, *graphs: "ExecutionGraph") -> "ExecutionGraph":
        """
        Combine multiple graphs into one.
        """
        combined_graph = cls()
        for graph in graphs:
            combined_graph = nx.compose(combined_graph, graph)

        return combined_graph

    def plot(self) -> None:
        """
        Plot the graph.
        """
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self)

        node_colors = [
            "green" if self.nodes[node].get("store_result", False) else "red"
            for node in self.nodes()
        ]

        nx.draw(
            self,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=2500,
            font_size=10,
        )

        plt.title("Expression Graph")
        plt.show()

    def set_sink_nodes_store(
        self, hops: int = 0, desired_category: CohortCategory | None = None
    ) -> None:
        """
        Set the store_result flag for all sink nodes of the graph.

        Find all nodes of the desired category that have no outgoing edges to nodes of the same category
        these are the last nodes of POPULATION, INTERVENTION or POPULATION_INTERVENTION and their result should
        be stored.

        :param hops: The number of hops to go back from the sink nodes.
        :param desired_category: The category of the sink nodes. None means all categories.
        """

        def is_sink_of_category(
            node: logic.Expr, graph: ExecutionGraph, category: CohortCategory
        ) -> bool:
            if graph.nodes[node]["category"] != category:
                return False  # Node is not of the desired category

            for _, neighbor in graph.out_edges(node):
                if graph.nodes[neighbor]["category"] == category:
                    return False  # Node has an outgoing edge to a node of the same category

            return True

        def set_predecessors_store(
            node: logic.Expr, graph: ExecutionGraph, hops_remaining: int
        ) -> None:
            if hops_remaining < 0:
                return
            for predecessor in graph.predecessors(node):
                if (
                    graph.nodes[predecessor]["category"]
                    == graph.nodes[node]["category"]
                ):
                    graph.nodes[predecessor]["store_result"] = True
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
                if is_sink_of_category(node, self, category)
            ]
            assert (
                len(sink_nodes_of_desired_category) <= 1
            ), "There must be zero or one sink nodes of the desired category"

            for sink_node in sink_nodes_of_desired_category:
                self.nodes[sink_node]["store_result"] = True
                set_predecessors_store(sink_node, self, hops)
