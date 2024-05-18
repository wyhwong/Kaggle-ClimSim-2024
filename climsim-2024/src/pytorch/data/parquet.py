import threading
from queue import Queue
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import torch

import src.core.workerpool
import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        source: str,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int = 3072,
        n_workers: int = 1,
        buffer_size: int = 10,  # 10 batches
        hold_on_time: float = 0.2,  # 0.2 seconds
        groups: Optional[list[int]] = None,
        n_batch_per_sampling: Optional[int] = None,
        n_group_per_sampling: Optional[int] = None,
        to_tensor: bool = True,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            source (str): The path to the source file
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used
            n_workers (int): The number of worker threads to be used
            buffer_size (int): The buffer size to be used
            hold_on_time (float): The time to wait before checking the buffer again (if the buffer is full)
            groups (list[int]): The groups to be used for sampling
            n_batch_per_sampling (int): The number of batches to be sampled per group
            n_group_per_sampling (int): The number of groups to be sampled per iteration
            to_tensor (bool): Convert the data to GPU tensors
            normalize (bool): Normalize the data

        Returns:
            None
        """

        self._parquet = pq.ParquetFile(source, memory_map=True, buffer_size=10)
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._batch_size = batch_size
        # If groups are not provided, use all the groups
        self._groups = groups or list(range(self._parquet.num_row_groups))
        # If n_batch_per_sampling is not provided, use 1 batch
        self._n_batch_per_sampling = n_batch_per_sampling or 1
        # If n_group_per_sampling is not provided, use all the groups
        self._n_group_per_sampling = n_group_per_sampling or self._parquet.num_row_groups
        self._n_workers = n_workers
        self._buffer_size = buffer_size
        self._hold_on_time = hold_on_time
        self._to_tensor = to_tensor
        self._normalize = normalize

        self._n_samples = sum([self._parquet.metadata.row_group(g).num_rows for g in self._groups])
        # Get some metadata from the source
        lf = pl.scan_parquet(source)
        # Get the number of samples (to calculate the length of the dataset)
        self._n_samples = lf.select(pl.len()).collect().item()
        if self._normalize:
            # Normalization for X
            self._X_min = np.array([lf.select(pl.min(col)).collect().item() for col in self._input_cols])
            self._X_scaling = (
                np.array([lf.select(pl.max(col)).collect().item() for col in self._input_cols]) - self._X_min
            )
            self._X_scaling[self._X_scaling == 0] = 1.0
            # Normalization for y
            self._y_min = np.array([lf.select(pl.min(col)).collect().item() for col in self._target_cols])
            self._y_scaling = (
                np.array([lf.select(pl.max(col)).collect().item() for col in self._target_cols]) - self._y_min
            )
            self._y_scaling[self._y_scaling == 0] = 1.0
        else:
            self._X_min = self._X_scaling = self._y_min = self._y_scaling = np.array([])

        self._shutdown_event = threading.Event()
        self._worker_pool = src.core.workerpool.WorkerPool()
        self._np_buffer: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=self._buffer_size)
        self._tensor_buffer: Queue[tuple[torch.Tensor, torch.Tensor]] = Queue(maxsize=self._buffer_size)

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index from the buffer"""

        if self._np_buffer.empty():
            local_logger.info("Buffer queue is empty. Waiting for batches to be loaded...")

        if self._to_tensor:
            return self._tensor_buffer.get()

        return self._np_buffer.get()

    def start_workers(self) -> None:
        """Start the worker threads"""

        for _ in range(self._n_workers):
            worker = threading.Thread(target=self._worker_fn, daemon=True)
            self._worker_pool.add_worker(worker=worker, start=True)

    def shutdown_workers(self) -> None:
        """Shutdown the threads"""

        self._shutdown_event.set()
        self._worker_pool.shutdown_workers()

    def _get_rows_group(self, size: int) -> np.ndarray:
        """Return a random group"""

        return np.random.choice(self._groups, size=size, replace=False)

    def _sample_from_df(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Sample from a DataFrame"""

        samples = df.sample(self._batch_size).reset_index(drop=True)
        X, y = samples[self._input_cols].values, samples[self._target_cols].values

        if self._normalize:
            return self._normalize_batch(X, y)
        return X, y

    def _normalize_batch(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize the data"""

        X = (X - self._X_min) / self._X_scaling
        y = (y - self._y_min) / self._y_scaling
        return X, y

    def _put_batch_to_tensor_buffer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Convert the data to GPU tensors"""

        _X = torch.Tensor(X).to(src.env.DEVICE)
        _y = torch.Tensor(y).to(src.env.DEVICE)
        self._tensor_buffer.put((_X, _y))

    def _worker_fn(self) -> None:
        """Load batches in the background and populate the buffer queue"""

        _pool = src.core.workerpool.WorkerPool()

        while True:
            if self._shutdown_event.is_set():
                local_logger.debug("Event has been set. Shutting down the worker...")
                _pool.shutdown_workers(1)
                return

            df = self._parquet.read_row_groups(
                row_groups=self._get_rows_group(self._n_group_per_sampling),
            ).to_pandas()

            for _ in range(self._n_batch_per_sampling):
                if self._shutdown_event.is_set():
                    break

                if self._np_buffer.full():
                    sleep(self._hold_on_time)
                    continue

                X, y = self._sample_from_df(df=df)

                if self._to_tensor:
                    worker = threading.Thread(target=self._put_batch_to_tensor_buffer, args=(X, y), daemon=True)
                    _pool.add_worker(worker=worker, start=True)
                else:
                    self._np_buffer.put((X, y))

            _pool.shutdown_workers()
            local_logger.debug("Cleaned up for sampling, waiting for the next iteration...")

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
