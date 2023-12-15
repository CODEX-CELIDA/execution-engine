import execution_engine.util.cohort_logic as logic
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.task.task import Task


class TaskCreator:
    """
    A TaskCreator object creates a Task tree for an expression and its dependencies.

    The TaskCreator object is used to create a Task tree for an expression and its dependencies.
    The Task tree is used to run the tasks in the correct order, obeying the dependencies.
    """

    def __init__(self, base_criterion: Criterion, params: dict | None = None) -> None:
        self.task_mapping: dict[logic.Expr | str, Task] = {}
        self.base_task = Task(
            expr=logic.Symbol(base_criterion.unique_name(), criterion=base_criterion),
            criterion=base_criterion,
            params=params,
            store_result=True,
        )
        self.params = params

    def create_tasks_and_dependencies(self, expr: logic.Expr) -> Task:
        """
        Creates a Task tree for an expression and its dependencies.

        :param expr: The expression to create a Task tree object for.
        :return: The (root) Task object for the expression.
        """

        def _create_subtasks(expr: logic.Expr) -> Task:
            if expr in self.task_mapping:
                return self.task_mapping[expr]

            current_criterion = expr.criterion if expr.is_Atom else None

            current_task = Task(
                expr=expr, criterion=current_criterion, params=self.params
            )

            dependencies = []
            if isinstance(expr, (logic.And, logic.Or, logic.Not)):
                for arg in expr.args:
                    child_task = _create_subtasks(arg)
                    dependencies.append(child_task)
            elif expr.is_Atom:
                # this task is a criterion and should store its result
                current_task.store_result = True
                dependencies.append(self.base_task)

            current_task.dependencies = dependencies

            self.task_mapping[expr] = current_task

            return current_task

        root_task = _create_subtasks(expr)
        root_task.store_result = True

        return root_task
