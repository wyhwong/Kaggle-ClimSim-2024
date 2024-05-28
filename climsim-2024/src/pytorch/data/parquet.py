import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import torch

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
        self._buffer_size = buffer_size
        self._hold_on_time = hold_on_time
        self._to_tensor = to_tensor
        self._normalize = normalize

        self._n_samples = sum([self._parquet.metadata.row_group(g).num_rows for g in self._groups])
        self._max_idx = self.__len__() - 1
        self._X_min = self._X_scaling = self._y_min = self._y_scaling = np.array([])
        self._init_scaling()

        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._sampling_thread = threading.Thread(target=lambda: None)
        self._np_buffer: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=self._buffer_size)
        self._tensor_buffer: Queue[tuple[torch.Tensor, torch.Tensor]] = Queue(maxsize=self._buffer_size)

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index from the buffer"""

        if idx == self._max_idx:
            local_logger.info("One epoch ended, Restarting the sampling thread...")
            self.shutdown_workers()
            self.start_workers()

        if self._np_buffer.empty() or self._tensor_buffer.empty():
            local_logger.debug("Buffer queue is empty. Waiting for batches to be loaded...")

        if self._to_tensor:
            return self._tensor_buffer.get()

        return self._np_buffer.get()

    def start_workers(self) -> None:
        """Start the worker threads"""

        self._lock.acquire()
        self._shutdown_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._worker_fn)
        self._sampling_thread.start()
        self._lock.release()
        local_logger.debug("Sampling thread has been started.")

    def shutdown_workers(self) -> None:
        """Shutdown the threads"""

        self._lock.acquire()
        self._shutdown_event.set()
        self._sampling_thread.join(timeout=1.0)
        self._lock.release()
        local_logger.debug("Sampling thread has been shut down.")

    def clean_up(self) -> None:
        """Clean up the resources"""

        self._np_buffer.queue.clear()
        self._tensor_buffer.queue.clear()

    def _init_scaling(self) -> None:
        """
        Get the scaling values for normalization

        NOTE: This function read the metadata of the source file to get the max and min values.
              It therefore is much faster than polars,
              which go through the whole file to get the max and min values.
        """

        cols = self._input_cols + self._target_cols
        max_values = {col: -np.inf for col in cols}
        min_values = {col: np.inf for col in cols}

        # Go through metadata of each row group to get the max and min values
        for row_group in range(self._parquet.num_row_groups):
            meta = self._parquet.metadata.row_group(row_group)
            for idx in range(meta.num_columns):
                col_meta = meta.column(idx)
                col = col_meta.path_in_schema
                if col in cols:
                    max_values[col] = max(max_values[col], col_meta.statistics.max)
                    min_values[col] = min(min_values[col], col_meta.statistics.min)

        ds_min, ds_max = pd.Series(min_values), pd.Series(max_values)

        self._X_min = ds_min[self._input_cols].values
        self._X_scaling = ds_max[self._input_cols].values - self._X_min
        self._X_scaling[self._X_scaling == 0] = 1.0

        self._y_min = ds_min[self._target_cols].values
        self._y_scaling = ds_max[self._target_cols].values - self._y_min
        self._y_scaling[self._y_scaling == 0] = 1.0

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

        _pool = ThreadPoolExecutor(max_workers=self._n_batch_per_sampling)
        futures: list[concurrent.futures._base.Future] = []

        while True:
            if self._shutdown_event.is_set():
                local_logger.debug("Event has been set. Shutting down the worker...")
                _pool.shutdown(wait=True, cancel_futures=True)
                return

            df = self._parquet.read_row_groups(
                row_groups=self._get_rows_group(self._n_group_per_sampling),
            ).to_pandas()
            wait(futures)
            futures.clear()

            for _ in range(self._n_batch_per_sampling):
                if self._shutdown_event.is_set():
                    break

                while psutil.virtual_memory().percent > 90.0:
                    local_logger.debug("Memory usage is high. Waiting for the memory to be freed...")
                    sleep(self._hold_on_time)

                while self._np_buffer.full() or self._tensor_buffer.full():
                    local_logger.debug("Buffer queue is full. Waiting for batches to be loaded...")
                    sleep(self._hold_on_time)

                X, y = self._sample_from_df(df=df)

                if self._to_tensor:
                    futures.append(_pool.submit(self._put_batch_to_tensor_buffer, X, y))
                else:
                    self._np_buffer.put((X, y))

            del df
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
