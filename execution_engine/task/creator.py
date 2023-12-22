import networkx as nx

import execution_engine.util.cohort_logic as logic
from execution_engine.task.task import Task


class TaskCreator:
    """
    A TaskCreator object creates a Task tree for an expression and its dependencies.

    The TaskCreator object is used to create a Task tree for an expression and its dependencies.
    The Task tree is used to run the tasks in the correct order, obeying the dependencies.
    """

    @staticmethod
    def create_tasks_and_dependencies(graph: nx.DiGraph) -> list[Task]:
        """
        Creates a Task tree for a graph and its dependencies.

        :return: The Graph object for the expression.
        """

        def node_to_task(node: logic.Expr, attr: dict) -> Task:
            criterion = attr["criterion"]

            task = Task(
                expr=node,
                criterion=criterion,
                params=attr["params"],
                store_result=attr["store_result"],
            )

            return task

        tasks: dict[logic.Expr, Task] = {
            node: node_to_task(node, attr) for node, attr in graph.nodes(data=True)
        }

        for node, task in tasks.items():
            predecessors = [tasks[pred] for pred in graph.predecessors(node)]
            task.dependencies = predecessors

        return list(tasks.values())
