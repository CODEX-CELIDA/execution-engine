import networkx as nx

import execution_engine.util.logic as logic
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

        def node_to_task(expr: logic.Expr, attr: dict) -> Task:
            store_result = attr.get("store_result", False)
            bind_params = attr.get("bind_params", {}).copy()
            bind_params["category"] = attr["category"]

            task = Task(
                expr=expr,
                bind_params=bind_params,
                store_result=store_result,
            )

            return task

        tasks: dict[logic.Expr, Task] = {
            node: node_to_task(node, attr) for node, attr in graph.nodes(data=True)
        }

        for node, task in tasks.items():
            predecessors = [tasks[pred] for pred in graph.predecessors(node)]
            task.dependencies = predecessors

        flattened_tasks = list(tasks.values())

        # we will make sure all tasks are depickled correctly
        for i, node in enumerate(tasks):
            if logic.Expr.from_dict(node.dict(include_id=True)) != node:
                raise RuntimeError(
                    "Expected depickled node to be the same as initial node."
                )

        assert (
            len(set(flattened_tasks))
            == len(flattened_tasks)
            == len(graph.nodes)
            == len(set([task.name() for task in flattened_tasks]))
        ), "Duplicate tasks found during task creation."

        return flattened_tasks
