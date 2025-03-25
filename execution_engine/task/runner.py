import logging
import multiprocessing
import pickle
import queue
import sys
import time
import traceback
from abc import ABC, abstractmethod
from multiprocessing.shared_memory import SharedMemory
from typing import (
    Callable,
    Dict,
    Generic,
    MutableMapping,
    Protocol,
    Tuple,
    TypeVar,
    cast,
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

    def enqueue_ready_tasks(self, bind_params: dict) -> int:
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
                logging.debug(f"Enqueuing task {task.name()}")

                # try:
                #     self.queue.put(task)
                # except Exception as ex:
                #     logging.error(f"Error enqueuing task {task.name()}: {ex}")
                #     raise ex
                #
                # with self.lock:
                #     self.enqueued_tasks.add(task.name())
                #     n_enqueued += 1
                with self.lock:
                    self._enqueue_task(task, bind_params)
                    self.enqueued_tasks.add(task.name())
                    n_enqueued += 1

        return n_enqueued

    def _enqueue_task(self, task: Task, bind_params: dict):
        """
        Actually put the item on a queue (abstract).
        Subclasses decide the exact structure to enqueue.
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, bind_params: dict) -> None:
        """
        Runs the tasks. This method must be implemented by subclasses.

        :param bind_params: The parameters to pass to the tasks.
        :return: None.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def shared_results(self) -> MutableMapping:
        """
        An abstract property that should be implemented by subclasses to return a dict-like object.
        """
        raise NotImplementedError()

    # @property
    # @abstractmethod
    # def queue(self) -> QueueLike[Task]:
    #     """
    #     An abstract property that should be implemented by subclasses to return a list-like object.
    #     """
    #
    # @property
    # def lock(self) -> ContextManager:
    #     """
    #     Returns a context manager for locking.
    #     By default, it's a no-op lock for single-threaded execution.
    #     Subclasses can override this with a real lock for multi-threaded execution.
    #     """
    #
    #     @contextmanager
    #     def no_op_lock() -> Iterator[None]:
    #         yield
    #
    #     return no_op_lock()
    #
    # def run_task(self, task: Task, bind_params: dict) -> None:
    #     """
    #     Runs a task.
    #
    #     :param task: The task to run.
    #     :param bind_params: The parameters to pass to the task.
    #     :return: None.
    #     """
    #     # Ensure all dependencies are processed first
    #     for dependency in task.dependencies:
    #         assert (
    #             dependency.name() in self.shared_results
    #         ), f"Dependency {dependency} is not finished."
    #
    #     input_data = [self.shared_results[dep.name()] for dep in task.dependencies]
    #     base_data = self.shared_results.get(task.get_base_task().name(), None)
    #
    #     result = task.run(input_data, base_data, bind_params)
    #
    #     with self.lock:
    #         self.shared_results[task.name()] = result


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
                    logging.debug(
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
        self._queue: multiprocessing.Queue = cast(
            multiprocessing.Queue, self.manager.Queue()
        )
        self._error_queue: multiprocessing.Queue = cast(
            multiprocessing.Queue, self.manager.Queue()
        )
        self.stop_event = self.manager.Event()
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

                logging.debug(f"Got task {task.name()}")

                try:
                    self.run_task(task, bind_params)

                    if task.status != TaskStatus.COMPLETED:
                        raise TaskError(
                            f"Task {task.name()} failed with status {task.status}"
                        )

                    logging.debug(f"Finished task {task.name()}")
                except TaskError as ex:
                    logging.error(ex)
                    self._error_queue.put(traceback.format_exc())
                    self.stop_event.set()
                except Exception as ex:
                    logging.error(f"Task {task.name()} failed: {ex}")
                    self._error_queue.put(traceback.format_exc())
                    self.stop_event.set()

            logging.info("Worker process stopped.")

        self.start_workers(task_executor_worker)

        task_names = {task.name() for task in self.tasks}

        try:
            while len(self.completed_tasks) < len(self.tasks):
                if self.stop_event.is_set():
                    break

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
                    n_completed = len(self.completed_tasks)
                    self.completed_tasks = set(self.shared_results.keys())
                    if len(self.completed_tasks) > n_completed:
                        logging.info(
                            f"Completed {len(self.completed_tasks)} of {len(self.tasks)} tasks"
                        )
                    if not all(task in task_names for task in self.completed_tasks):
                        raise TaskError(
                            "Completed tasks differ from actual tasks "
                            "- problem with pickling/unpickling in multiprocessing?"
                        )

        except Exception as e:
            logging.error(f"An error occurred: {e}")
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

        if not self._error_queue.empty():
            logging.error("Errors occurred during task execution.")
            self.print_errors()
            self.manager.shutdown()
            sys.exit(1)

    def print_errors(self) -> None:
        """
        Print errors from the error queue.

        :return: None.
        """

        while not self._error_queue.empty():
            error_trace = self._error_queue.get()
            print(error_trace, file=sys.stderr)


from multiprocessing import resource_tracker


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# -----------------------------------------------------------------------
# Helper: store data into shared memory, return (shm_name, size)
# -----------------------------------------------------------------------
def store_in_shared_memory(data: PersonIntervals) -> Tuple[str, int]:
    """
    Example: we pickle the data to bytes, then copy it into a SharedMemory block.
    Returns (shm_name, shm_size). You can do a more sophisticated layout if you like.
    """
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    size = len(serialized)

    # Create a brand new shared memory block big enough to hold 'serialized'.
    shm = SharedMemory(create=True, size=size)
    try:
        # Copy into shm.buf
        shm.buf[:size] = serialized
        # Return the name and total size for the consumer
        return shm.name, size
    except Exception:
        # If anything fails, we should cleanup to avoid a memory leak
        shm.close()
        shm.unlink()
        raise


# -----------------------------------------------------------------------
# Helper: load data from a shared memory block by name/size
# -----------------------------------------------------------------------
def load_from_shared_memory(shm_name: str, shm_size: int):
    existing_shm = SharedMemory(name=shm_name, create=False)
    try:
        # Make an actual copy into a new bytes object:
        data_bytes = bytes(existing_shm.buf[:shm_size])
        data = pickle.loads(data_bytes)
    finally:
        # Now it is safe to close, because we no longer hold a memoryview
        existing_shm.close()
    return data


# -----------------------------------------------------------------------
class SharedMemoryTaskRunner(TaskRunner):
    def __init__(self, execution_graph: ExecutionGraph, num_workers: int = 4):
        super().__init__(execution_graph)
        if num_workers < 1:
            num_workers = multiprocessing.cpu_count()

        self.num_workers = num_workers
        self._lock = multiprocessing.Lock()

        # local results dictionary only in the main process
        self._shared_results: Dict[str, PersonIntervals] = {}

        # Two queues: tasks in, results out
        self._task_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()

        self.stop_event = multiprocessing.Event()
        self.workers = []
        self._shared_memory_blocks = []
        logging.info(f"Using SharedMemoryTaskRunner with {num_workers} workers.")

    @property
    def lock(self):
        return self._lock

    @property
    def shared_results(self) -> MutableMapping[str, PersonIntervals]:
        return self._shared_results

    def _enqueue_task(self, task: Task, bind_params: dict):
        # We do not store partial results in a Manager dict.
        # Instead, we send minimal info to the worker:
        #   - the task itself
        #   - the raw input data / base_data (which we gather in the main process)
        deps_data = [self._shared_results[dep.name()] for dep in task.dependencies]
        base_data = self._shared_results.get(task.get_base_task().name(), None)

        # Put everything needed onto the queue
        self._task_queue.put((task, deps_data, base_data, bind_params))

    def run(self, bind_params: dict) -> None:
        self.start_workers()

        try:
            total = len(self.tasks)
            while len(self.completed_tasks) < total:
                # Enqueue newly-ready tasks:
                self.enqueue_ready_tasks(bind_params)

                # If no tasks can be run, but not done => error
                if (
                    self.completed_tasks == self.enqueued_tasks
                    and len(self.completed_tasks) < total
                ):
                    raise TaskError("No runnable tasks available.")

                time.sleep(0.05)

                # Collect any finished results from the worker(s)
                self.fetch_results()

        except Exception as e:
            logging.error(f"Error in SharedMemoryTaskRunner: {e}")
            self.stop_event.set()
        finally:
            self.stop_workers()

    def fetch_results(self):
        """
        Drain the result queue. Each item is (task_name, shm_name, shm_size, error).
        If error is not None, the task crashed.
        Otherwise we load from shared memory, store in _shared_results.
        """
        try:
            while True:
                task_name, shm_name, shm_size, error = self._result_queue.get_nowait()
                self._shared_memory_blocks.append(shm_name)
                if error:
                    # The worker signaled an error
                    logging.error(f"Task {task_name} failed:\n{error}")
                    self.stop_event.set()
                    raise TaskError(f"Task {task_name} crashed: {error}")

                # Otherwise, load from shared memory
                result_data = load_from_shared_memory(shm_name, shm_size)

                # (Optional) unlink if we know we won’t need it again
                # BUT be sure no one else is reading it, etc.
                #    existing_shm = SharedMemory(shm_name)
                #    existing_shm.unlink()

                # Store in local dictionary
                with self.lock:
                    self._shared_results[task_name] = result_data
                    self.completed_tasks.add(task_name)

        except queue.Empty:
            return

    def start_workers(self):
        """
        Start worker processes that read from _task_queue until a sentinel is received.
        """

        def worker_loop(task_queue, result_queue, stop_ev):
            worker_shm = []

            while not stop_ev.is_set():
                try:
                    item = task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    # sentinel
                    break

                task, deps_data, base_data, bp = item

                try:
                    # Actually run the task
                    result = task.run(deps_data, base_data, bp)
                    if task.status != TaskStatus.COMPLETED:
                        raise TaskError(f"Task {task.name()} ended with {task.status}")

                    # Now store the large result in shared memory
                    shm_name, shm_size = store_in_shared_memory(result)

                    worker_shm.append(shm_name)

                    # Send back just a small descriptor
                    result_queue.put((task.name(), shm_name, shm_size, None))

                except Exception as ex:
                    tb = traceback.format_exc()
                    logging.error(f"Task {task.name()} crashed with {ex}:\n{tb}")
                    result_queue.put((task.name(), "", 0, tb))
                    stop_ev.set()
            # cleanup
            logging.info("Cleanup")

            for shm_name in worker_shm:
                # print(shm_name)
                existing_shm = SharedMemory(name=shm_name, create=False)
                existing_shm.close()
                existing_shm.unlink()

        logging.info(f"Starting {self.num_workers} processes.")
        self.stop_event.clear()
        for _ in range(self.num_workers):
            p = multiprocessing.Process(
                target=worker_loop,
                args=(self._task_queue, self._result_queue, self.stop_event),
            )
            p.start()
            self.workers.append(p)

    def stop_workers(self):
        logging.info("Stopping worker processes.")

        for shm_name in self._shared_memory_blocks:
            existing_shm = SharedMemory(name=shm_name, create=False)
            existing_shm.close()
            # existing_shm.unlink()
        logging.info("Cleaned up shared memory")

        for _ in self.workers:
            self._task_queue.put(None)  # sentinel
        for w in self.workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
                w.join()

        logging.info("All worker processes stopped.")
