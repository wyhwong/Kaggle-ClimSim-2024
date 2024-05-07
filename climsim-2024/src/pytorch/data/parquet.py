import threading
from queue import Queue
from time import sleep

import numpy as np
import pyarrow.parquet as pq
import torch

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        parquet: pq.ParquetFile,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
        n_samples: int,
        groups: list[int],
        n_batch_per_sampling: int = 3,
        n_group_per_sampling: int = 2,
        n_workers: int = 1,
        buffer_size: int = 10,
        hold_on_time: float = 0.2,
        to_tensor: bool = True,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            parquet (pq.ParquetFile): The dataset to be used
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used
            n_samples (int): The number of samples in the dataset
            groups (list[int]): The groups to be used for sampling
            n_batch_per_sampling (int): The number of batches to be sampled per group
            n_group_per_sampling (int): The number of groups to be sampled per iteration
            n_workers (int): The number of worker threads to be used
            buffer_size (int): The buffer size to be used
            hold_on_time (float): The time to wait before checking the buffer again (if the buffer is full)
            to_gpu (bool): Whether to convert the data to GPU tensors

        Returns:
            None
        """

        self._parquet = parquet
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._batch_size = batch_size
        self._n_samples = n_samples
        self._buffer_size = buffer_size
        self._n_batch_per_sampling = n_batch_per_sampling
        self._n_group_per_sampling = n_group_per_sampling
        self._hold_on_time = hold_on_time
        self._shutdown_event = threading.Event()
        self._to_tensor = to_tensor

        self._X: torch.Tensor = torch.Tensor([])
        self._y: torch.Tensor = torch.Tensor([])
        self._get_group_fn = lambda: np.random.choice(groups)

        self._buffer_queue: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=self._buffer_size)
        self._gpu_buffer_queue: Queue[tuple[torch.Tensor, torch.Tensor]] = Queue(maxsize=self._buffer_size)
        self._threads = []

        for idx in range(n_workers):
            self._threads.append(threading.Thread(target=self._load_batches, daemon=True))
            self._threads[idx].start()

    def _load_batches(self) -> None:
        """Load batches in the background and populate the buffer queue"""

        while True:
            if self._buffer_queue.full():
                sleep(self._hold_on_time)
                continue

            if self._shutdown_event.is_set():
                break

            df = self._parquet.read_row_groups(
                row_groups=[self._get_group_fn() for _ in range(self._n_group_per_sampling)],
                use_threads=True,
            ).to_pandas()

            for _ in range(self._n_batch_per_sampling):
                if self._buffer_queue.full():
                    break

                samples = df.sample(self._batch_size).reset_index(drop=True)
                X, y = samples[self._input_cols].values, samples[self._target_cols].values
                if self._to_tensor:
                    threading.Thread(target=self._to_gpu, args=(X, y), daemon=True).start()
                else:
                    self._buffer_queue.put((X, y))

    def _to_gpu(self, X: np.ndarray, y: np.ndarray) -> None:
        """Convert the data to GPU tensors"""

        _X = torch.Tensor(X).to(src.env.DEVICE)
        _y = torch.Tensor(y).to(src.env.DEVICE)
        self._gpu_buffer_queue.put((_X, _y))

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index from the buffer"""

        if self._buffer_queue.empty():
            local_logger.info("Buffer queue is empty. Waiting for batches to be loaded...")

        if self._to_tensor:
            return self._gpu_buffer_queue.get()

        return self._buffer_queue.get()

    def shutdown(self) -> None:
        """Shutdown the threads"""

        self._shutdown_event.set()
        for thread in self._threads:
            thread.join()

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a torch DataLoader object"""

        return torch.utils.data.DataLoader(
            self,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(self),
                batch_size=1,
                drop_last=False,
            ),
            **kwargs,
        )
