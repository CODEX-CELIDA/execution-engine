import logging
import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Callable,
    ContextManager,
    Generic,
    Iterator,
    MutableMapping,
    Protocol,
    TypeVar,
)

from execution_engine.execution_graph import ExecutionGraph
from execution_engine.task.creator import TaskCreator
from execution_engine.task.task import Task, TaskError, TaskStatus
from execution_engine.util.types import PersonIntervals

T = TypeVar("T")


class QueueLike(Protocol, Generic[T]):
    """
    A protocol for queue-like objects.
    """

    def put(self, item: T) -> None:
        """
        Put an item into the queue.
        """
        ...

    def get(self) -> T:
        """
        Get an item from the queue.
        """
        ...


def flatten_tasks(execution_graph: ExecutionGraph) -> list[Task]:
    """
    Flattens a task graph into a list of tasks.

    :param execution_graph: The execution graph.
    :return: A list of tasks.
    """
    return execution_graph.nodes()


class TaskRunner(ABC):
    """
    An abstract class for running a list of tasks.
    """

    def __init__(self, execution_graph: ExecutionGraph):
        self.tasks = TaskCreator.create_tasks_and_dependencies(execution_graph)
        self.completed_tasks: set[str] = set()
        self.enqueued_tasks: set[str] = set()

        logging.info(f"Scheduling {len(self.tasks)} tasks")

    def enqueue_ready_tasks(self) -> int:
        """
        Enqueues tasks that are ready to run based on their dependencies and completion status.

        :return: The number of tasks enqueued.
        """

        n_enqueued = 0

        for task in self.tasks:
            if (
                task.name() not in self.completed_tasks
                and all(dep.name() in self.completed_tasks for dep in task.dependencies)
                and task.name() not in self.enqueued_tasks
            ):
                logging.info(f"Enqueuing task {task.name()}")

                try:
                    self.queue.put(task)
                except Exception as ex:
                    logging.error(f"Error enqueuing task {task.name()}: {ex}")
                    raise ex

                with self.lock:
                    self.enqueued_tasks.add(task.name())
                    n_enqueued += 1

        return n_enqueued

    @abstractmethod
    def run(self, bind_params: dict) -> None:
        """
        Runs the tasks. This method must be implemented by subclasses.

        :param bind_params: The parameters to pass to the tasks.
        :return: None.
        """

    @property
    @abstractmethod
    def shared_results(self) -> MutableMapping:
        """
        An abstract property that should be implemented by subclasses to return a dict-like object.
        """

    @property
    @abstractmethod
    def queue(self) -> QueueLike[Task]:
        """
        An abstract property that should be implemented by subclasses to return a list-like object.
        """

    @property
    def lock(self) -> ContextManager:
        """
        Returns a context manager for locking.
        By default, it's a no-op lock for single-threaded execution.
        Subclasses can override this with a real lock for multi-threaded execution.
        """

        @contextmanager
        def no_op_lock() -> Iterator[None]:
            yield

        return no_op_lock()

    def run_task(self, task: Task, bind_params: dict) -> None:
        """
        Runs a task.

        :param task: The task to run.
        :param bind_params: The parameters to pass to the task.
        :return: None.
        """
        # Ensure all dependencies are processed first
        for dependency in task.dependencies:
            assert (
                dependency.name() in self.shared_results
            ), f"Dependency {dependency} is not finished."

        input_data = [self.shared_results[dep.name()] for dep in task.dependencies]
        base_data = self.shared_results.get(task.get_base_task().name(), None)

        result = task.run(input_data, base_data, bind_params)

        with self.lock:
            self.shared_results[task.name()] = result


class SequentialTaskRunner(TaskRunner):
    """
    Runs a list of tasks sequentially.
    """

    def __init__(self, execution_graph: ExecutionGraph):
        super().__init__(execution_graph)

        logging.info("Using sequential task runner.")

        self._shared_results: dict[str, PersonIntervals] = {}
        self._queue: queue.Queue = queue.Queue()

    @property
    def shared_results(self) -> MutableMapping:
        """
        Returns the shared results.
        """
        return self._shared_results

    @property
    def queue(self) -> queue.Queue:
        """
        Returns the queue of tasks to run.
        """
        return self._queue

    def run(self, bind_params: dict) -> None:
        """
        Runs the tasks sequentially.

        :param bind_params: The parameters to pass to the tasks.
        :return: None.
        """
        try:
            while len(self.completed_tasks) < len(self.tasks):
                self.enqueue_ready_tasks()

                if self.queue.empty() and not any(
                    task.status == TaskStatus.RUNNING for task in self.tasks
                ):
                    raise TaskError("No runnable tasks available.")

                while not self.queue.empty():
                    current_task = self.queue.get()
                    self.run_task(current_task, bind_params)

                    if current_task.status != TaskStatus.COMPLETED:
                        raise TaskError(
                            f"Task {current_task.name()} failed with status {current_task.status}"
                        )

                with self.lock:
                    # Update the set of completed tasks
                    self.completed_tasks = set(self.shared_results.keys())
                    logging.info(
                        f"Completed {len(self.completed_tasks)} of {len(self.tasks)} tasks"
                    )

        except TaskError as e:
            logging.error(f"An error occurred: {e}")
            logging.error("Stopping task runner.")
            raise TaskError(str(e))


class ParallelTaskRunner(TaskRunner):
    """
    Runs a list of tasks in parallel.

    :param num_workers: The number of worker processes to use. If -1, the number of CPUs is used.
    """

    def __init__(self, execution_graph: ExecutionGraph, num_workers: int = 4):
        super().__init__(execution_graph)

        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()

        logging.info(f"Using parallel task runner with {num_workers} workers.")

        self.num_workers = num_workers
        self.manager = multiprocessing.Manager()
        self._shared_results = self.manager.dict()
        self._lock = self.manager.Lock()
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.workers: list[multiprocessing.Process] = []

    @property
    def shared_results(self) -> MutableMapping:
        """
        Returns the shared results.
        """
        return self._shared_results

    @property
    def queue(self) -> multiprocessing.Queue:
        """
        Returns the queue of tasks to run.
        """
        return self._queue

    def run(self, bind_params: dict) -> None:
        """
        Runs the tasks in parallel.

        :param bind_params: The parameters to pass to the tasks.
        :return: None.
        """

        def task_executor_worker() -> None:
            """
            A worker process that executes tasks from a queue.

            :return: None.
            """

            while not self.stop_event.is_set():
                try:
                    task = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue  # Go back to check stop_event
                except Exception as ex:
                    logging.error(f"Error getting task from queue: {ex}")
                    self.stop_event.set()
                    continue

                logging.info(f"Got task {task.name()}")

                try:
                    self.run_task(task, bind_params)

                    if task.status != TaskStatus.COMPLETED:
                        raise TaskError(
                            f"Task {task.name()} failed with status {task.status}"
                        )

                    logging.info(f"Finished task {task.name()}")
                except TaskError as ex:
                    logging.error(ex)
                    self.stop_event.set()
                except Exception as ex:
                    logging.error(f"Task {task.name()} failed: {ex}")
                    self.stop_event.set()

            logging.info("Worker process stopped.")

        self.start_workers(task_executor_worker)

        try:
            while len(self.completed_tasks) < len(self.tasks):
                if self.stop_event.is_set():
                    raise TaskError("Task execution failed.")

                self.enqueue_ready_tasks()

                if self.completed_tasks == self.enqueued_tasks and len(
                    self.completed_tasks
                ) < len(self.tasks):
                    logging.info(
                        f"# Completed tasks: {len(self.completed_tasks)}, # Tasks: {len(self.tasks)}, # Enqueued tasks: {len(self.enqueued_tasks)}"
                    )
                    raise TaskError("No runnable tasks available.")

                time.sleep(0.1)

                with self.lock:
                    # Update the set of completed tasks
                    self.completed_tasks = set(self.shared_results.keys())

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise TaskError(str(e))
        finally:
            self.stop_workers()

    def start_workers(self, func: Callable) -> None:
        """
        Starts worker processes.

        :param func: The function to run in each worker process.
        :return: None.
        """

        logging.info(f"Starting {self.num_workers} worker processes.")

        self.stop_event.clear()

        self.workers = [
            multiprocessing.Process(target=func)
            for worker_index in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()

    def stop_workers(self, timeout: int = 10) -> None:
        """
        Stops worker processes.

        :param timeout: The timeout in seconds to wait for the worker processes to finish.
        :return: None.
        """
        logging.info("Stopping worker processes")

        self.stop_event.set()

        for w in self.workers:
            w.join(timeout=timeout)
            if w.is_alive():
                logging.warning(
                    f"Worker {w.name} did not finish within timeout. Terminating..."
                )
                w.terminate()
                w.join()

        logging.info("All worker processes stopped.")
