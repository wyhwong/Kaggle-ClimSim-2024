import threading
from queue import Queue

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.utils

import src.env
import src.logger
import src.schemas.math


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
        n_batch_per_sampling: int = 2,
        buffer_size: int = 5,
        num_workers: int = 2,
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
            buffer_size (int): The buffer size to be used

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

        self._X: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])
        self._get_group_fn = lambda: np.random.choice(groups)

        self._buffer_queue: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=self._buffer_size)

        _n_threads = 0
        while _n_threads < num_workers:
            _worker_thread = threading.Thread(target=self._load_batches, daemon=True)
            _worker_thread.start()
            _n_threads += 1

    def _load_batches(self) -> None:
        """Load batches in the background and populate the buffer queue"""

        while True:
            if self._buffer_queue.full():
                continue

            df = self._parquet.read_row_group(self._get_group_fn()).to_pandas()
            for _ in range(self._n_batch_per_sampling):
                if self._buffer_queue.full():
                    break

                samples = df.sample(self._batch_size)
                X, y = samples[self._input_cols].values, samples[self._target_cols].values
                self._buffer_queue.put((X, y))

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, _) -> tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index from the buffer"""

        if self._buffer_queue.empty():
            local_logger.warning("Buffer queue is empty. Waiting for batches to be loaded...")

        X, y = self._buffer_queue.get()
        return (X, y)

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a torch DataLoader object"""

        return torch.utils.data.DataLoader(
            self,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(self),
                batch_size=1,
                drop_last=False,
            ),
            pin_memory=True,
            **kwargs,
        )
