import threading
from typing import Optional

import src.logger


local_logger = src.logger.get_logger(__name__)


class WorkerPool:
    """A WorkerPool class for managing worker threads"""

    def __init__(self) -> None:
        """Initialize the WorkerPool"""

        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()

    def add_worker(self, worker: threading.Thread, start: bool = True) -> None:
        """
        Add a worker thread to the pool

        Args:
            worker (threading.Thread): The worker thread to be added

        Returns:
            None
        """

        local_logger.debug("Adding a worker thread to the pool...")

        with self._lock:
            self._workers.append(worker)
            if start:
                self._workers[-1].start()
                local_logger.debug("Started a worker thread.")

        local_logger.debug("Worker thread has been added to the pool.")

    def start_workers(self) -> None:
        """Start all the worker threads"""

        local_logger.debug("Starting all worker threads...")

        with self._lock:
            for worker in self._workers:
                worker.start()

        local_logger.debug("All worker threads have been started.")

    def shutdown_workers(self, timeout: Optional[float] = None) -> None:
        """Shutdown all the worker threads"""

        local_logger.debug("Shutting down all worker threads...")

        with self._lock:
            while self._workers:
                self._workers.pop().join(timeout)

        local_logger.debug("All worker threads have been shutdown.")
