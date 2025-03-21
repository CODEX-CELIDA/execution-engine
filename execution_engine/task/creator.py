import pickle  # nosec

import networkx as nx
from typing_extensions import Any

import execution_engine.util.logic as logic
from execution_engine.task.task import Task


def assert_pickle_roundtrip(obj: logic.BaseExpr) -> None:
    """
    Serializes 'obj' via pickle (the same method multiprocessing would use),
    then deserializes it, and finally compares the original object to the result.

    :param obj: The object to serialize/deserialize.
    :raises AssertionError: If the object does not match its clone.
    :return: The deserialized clone (for further inspection if needed).
    """
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)  # nosec
    clone = pickle.loads(pickled)  # nosec

    if isinstance(obj, logic.CountOperator):
        if obj.dict()["data"]["threshold"] is None:
            raise AssertionError("Threshold must be set")

    if obj == clone:
        return  # They are considered equal, so nothing to do.

    # If they're unequal, compare their dict() representations to find differences.
    d1 = obj.dict()
    d2 = clone.dict()

    if d1 == d2:
        # If they're unequal but dicts are the same, there's some internal difference
        # not visible via .dict(). Just alert that we can't show details.
        raise AssertionError(
            f"Objects differ in __eq__, but their dict() representations are identical.\n"
            f"obj:   {obj}\n"
            f"clone: {clone}"
        )

    # Otherwise, gather all leaf-level differences in d1 vs d2.
    diffs = _compare_dicts_leaf_level(d1, d2)
    diff_msg = "\n".join(diffs)

    raise AssertionError(
        f"Object does not match its clone after round-trip!\n\n"
        f"Differences at leaf level in .dict() representations:\n{diff_msg}"
    )


def _compare_dicts_leaf_level(d1: Any, d2: Any, path: str = "") -> list[str]:
    """
    Recursively compare two dict/list/tuple/scalar structures and return
    a list of strings describing differences at the leaf level.

    :param d1, d2: Potentially nested structures (dict, list, tuple, scalar).
    :param path: Path string to locate the current point in the structure.
    :return: List of difference descriptions.
    """
    differences = []

    # If both are dicts, recurse into matching keys
    if isinstance(d1, dict) and isinstance(d2, dict):
        all_keys = set(d1.keys()) | set(d2.keys())
        for key in sorted(all_keys):
            sub_path = f"{path}.{key}" if path else str(key)
            if key not in d1:
                differences.append(f"[MISSING IN ORIGINAL] {sub_path} => {d2[key]!r}")
            elif key not in d2:
                differences.append(f"[MISSING IN CLONE] {sub_path} => {d1[key]!r}")
            else:
                differences.extend(
                    _compare_dicts_leaf_level(d1[key], d2[key], sub_path)
                )

    # If both are lists/tuples, compare element by element
    elif isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        if len(d1) != len(d2):
            differences.append(f"[LEN MISMATCH] {path} => {len(d1)} vs {len(d2)}")
        else:
            for i, (item1, item2) in enumerate(zip(d1, d2)):
                sub_path = f"{path}[{i}]"
                differences.extend(_compare_dicts_leaf_level(item1, item2, sub_path))

    # Otherwise, treat them as leaf values and compare directly
    else:
        if d1 != d2:
            differences.append(f"[VALUE MISMATCH] {path} => {d1!r} vs {d2!r}")

    return differences


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

        # we will make sure all tasks are depickled correctly [commented out for performance reasons]
        # from tqdm import tqdm
        #
        # for node in tqdm(tasks):
        #     assert_pickle_roundtrip(node)

        assert (
            len(set(flattened_tasks))
            == len(flattened_tasks)
            == len(graph.nodes)
            == len(set([task.name() for task in flattened_tasks]))
        ), "Duplicate tasks found during task creation."

        return flattened_tasks
