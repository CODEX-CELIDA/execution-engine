import logging
import multiprocessing
import time
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Lock as MPLock
from types import TracebackType
from typing import Type

import pandas as pd

from execution_engine.task import Task, TaskState


class MockLock:
    """
    A mock lock that does nothing.

    Used when processing tasks sequentially.
    """

    def acquire(self) -> None:
        """
        Acquires the lock (does nothing).
        """

    def release(self) -> None:
        """
        Releases the lock (does nothing).
        """

    def __enter__(self) -> "MockLock":
        """
        Context manager enter (does nothing).
        """
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Context manager exit (does nothing).
        """


def flatten_tasks(root_task: Task) -> list[Task]:
    """
    Flattens a task tree into a list of tasks.

    :param root_task: The root task to flatten.
    :return: A list of tasks.
    """
    tasks = []
    queue = [root_task]
    while queue:
        current_task = queue.pop()
        if current_task not in tasks:
            tasks.append(current_task)
        queue.extend(current_task.dependencies)
    return tasks


def run_task(
    task: Task, shared_results: dict | DictProxy, lock: MPLock | MockLock
) -> None:
    """
    Runs a task.

    :param task: The task to run.
    :param shared_results: A shared dictionary of results.
    :param lock: A lock to synchronize access to the shared results.
    :return: None.
    """
    # Base case: if the task is already processed, return
    if task.is_finished():
        return

    # Ensure all dependencies are processed first
    for dependency in task.dependencies:
        assert (
            dependency.name() in shared_results
        ), f"Dependency {dependency} is not finished."

    input_data = [shared_results[dep.name()] for dep in task.dependencies]

    result = task.run(input_data)

    with lock:
        shared_results[task.name()] = result

    # Check if the task is finished and handle accordingly
    if not task.is_finished():
        logging.error(f"Task {task} failed to complete.")


def run_tasks_sequential(tasks: list[Task]) -> None:
    """
    Runs a list of tasks sequentially.

    :param tasks: A list of tasks to run.
    :return: None.
    """
    queue = []
    shared_results: dict[str, pd.DataFrame] = {}
    lock = MockLock()

    def enqueue_ready_tasks() -> None:
        """
        Enqueues tasks that are ready to run.
        """
        for task in tasks:
            if task.state == TaskState.NOT_RUN and all(
                dep.is_finished() for dep in task.dependencies
            ):
                queue.append(task)

    while not all(task.is_finished() for task in tasks):
        enqueue_ready_tasks()
        while len(queue):
            current_task = queue.pop(0)
            run_task(current_task, shared_results, lock)


def task_executor(
    task_queue: multiprocessing.Queue, shared_results: DictProxy, lock: MPLock
) -> None:
    """
    A worker process that executes tasks from a queue.

    :param task_queue: A queue of tasks to execute.
    :param shared_results: A shared dictionary of results.
    :param lock: A lock to synchronize access to the shared results.
    :return: None.
    """

    while True:
        task = task_queue.get()

        logging.info(f"Got task {task}")

        if task is None:  # Poison pill means shutdown
            task_queue.put(None)  # Put it back for other processes
            break

        run_task(task, shared_results, lock)

        logging.info(f"Finished task {task}")


def run_tasks_parallel(tasks: list[Task], num_workers: int = 4) -> None:
    """
    Runs a list of tasks in parallel.

    :param tasks: A list of tasks to run.
    :param num_workers: The number of workers to use.
    :return: None.
    """
    manager = multiprocessing.Manager()
    shared_results = manager.dict()
    lock = manager.Lock()

    task_queue: multiprocessing.Queue = multiprocessing.Queue()

    # Start worker processes
    workers = [
        multiprocessing.Process(
            target=task_executor, args=(task_queue, shared_results, lock)
        )
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    completed_tasks: set[str] = set()
    enqueued_tasks: set[str] = set()

    while len(completed_tasks) < len(tasks):
        for task in tasks:
            if (
                task.name() not in completed_tasks
                and all(dep.name() in completed_tasks for dep in task.dependencies)
                and task.name() not in enqueued_tasks
            ):
                logging.info(f"Enqueuing task {task.name()}")
                task_queue.put(task)
                enqueued_tasks.add(task.name())

        time.sleep(0.1)

        with lock:
            completed_tasks = set(
                shared_results.keys()
            )  # Update the set of completed tasks

    # Tell child processes to stop
    task_queue.put(None)
    for w in workers:
        w.join()

    # Access results from shared_results
    # for task_name, result in shared_results.items():
    #    print(f"Results for {task_name}: {result}")
