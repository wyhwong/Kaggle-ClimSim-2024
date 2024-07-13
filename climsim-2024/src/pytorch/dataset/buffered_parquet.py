import threading
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
import src.pytorch.dataset.base as base


local_logger = src.logger.get_logger(__name__)


class BufferedParquetDataset(base.DatasetBase):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        source: str,
        x_stats: str,
        y_stats: str,
        batch_size: int = 4096,
        buffer_size: int = 100,
        hold_on_time: float = 0.2,  # 0.2 seconds
        n_group_per_sampling: Optional[int] = None,
        groups: Optional[list[int]] = None,
        device: str = "cuda",
        is_to_tensor: bool = True,
        is_normalize: bool = False,
        is_standardize: bool = True,
    ) -> None:
        """BufferedParquetDataset constructor

        Args:
            source (str): The path to the dataset
            x_stats (str): The path to the x statistics (parquet file)
            y_stats (str): The path to the y statistics (parquet file)
            batch_size (int): The batch size
            buffer_size (int): The buffer size
            hold_on_time (float): The time to hold on
            n_group_per_sampling (int): The number of groups per sampling
            groups (list[int]): The groups
            device (str): The device to use
            is_to_tensor (bool): Whether to convert to tensor
            is_normalize (bool): Whether to normalize the data
            is_standardize (bool): Whether to standardize the data

        Methods (excluding inherited methods):
            __len__: Return the length of the dataset (unit in batch)
            __getitem__: Return the data at the given index from the buffer
            start_sampling_worker: Start the worker threads
            shutdown_sampling_worker: Shutdown the threads
            clean_up: Clean up the resources
            _get_rows_group: Return a random group
            _sample_from_df: Sample from a DataFrame
            _sampling_worker_fn: Load batches in the background and populate the buffer queue
            _sleep_if_memory_high: Sleep if the memory usage is high
            _sleep_if_buffer_full: Sleep if the buffer queue is full
            generate_tiny_dataset: Generate a tiny dataset
            get_batch: Get a batch of data
            to_dataloader: Return a torch DataLoader object
        """

        super().__init__(source, x_stats, y_stats, device, is_to_tensor, is_normalize, is_standardize)

        self._batch_size = batch_size
        self._hold_on_time = hold_on_time

        self._buffer_size = buffer_size
        self._buffer: Queue[tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]] = Queue(
            maxsize=self._buffer_size
        )

        self._parquet = pq.ParquetFile(self._source, memory_map=True, buffer_size=10)
        # If n_group_per_sampling is not provided, use all the groups
        self._n_group_per_sampling = n_group_per_sampling or self._parquet.num_row_groups
        # If groups are not provided, use all the groups
        self._groups = groups or list(range(self._parquet.num_row_groups))

        self._n_samples = sum([self._parquet.metadata.row_group(g).num_rows for g in self._groups])
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._sampling_thread = threading.Thread(target=lambda: None, daemon=True)

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, idx: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Return the data at the given index from the buffer"""

        if not self._sampling_thread.is_alive():
            self._start_sampling_worker()

        if idx + 1 == self.__len__():
            self._shutdown_sampling_worker()
            self._start_sampling_worker()

        if self._buffer.empty():
            local_logger.debug("Buffer queue is empty. Waiting for batches to be loaded...")

        return self._buffer.get()

    def _start_sampling_worker(self) -> None:
        """Start the worker threads"""

        self._lock.acquire()
        self._shutdown_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sampling_worker_fn, daemon=True)
        self._sampling_thread.start()
        self._lock.release()
        local_logger.debug("Sampling thread has been started.")

    def _shutdown_sampling_worker(self) -> None:
        """Shutdown the threads"""

        self._lock.acquire()
        self._shutdown_event.set()
        self._sampling_thread.join()
        self._lock.release()
        local_logger.debug("Sampling thread has been shut down.")

    def clean_up(self) -> None:
        """Clean up the resources"""

        if self._sampling_thread.is_alive():
            self._shutdown_sampling_worker()

        self._buffer.queue.clear()

    def _get_rows_group(self, size: int) -> np.ndarray:
        """Return a random group"""

        return np.random.choice(self._groups, size=size, replace=False)

    def _sample_from_df(
        self, df: pd.DataFrame, batch_size: Optional[int] = None
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Sample from a DataFrame"""

        batch_size = batch_size or self._batch_size

        samples = df.sample(batch_size)
        x, y = samples[self.input_cols].values, samples[self.output_cols].values
        return self.preprocess_features(x), self.preprocess_targets(y)

    def _sampling_worker_fn(self) -> None:
        """Load batches in the background and populate the buffer queue"""

        while True:
            if self._shutdown_event.is_set():
                local_logger.debug("Event has been set. Shutting down the worker...")
                return

            df = self._parquet.read_row_groups(
                row_groups=self._get_rows_group(self._n_group_per_sampling),
            ).to_pandas()
            local_logger.debug("Loaded the row groups for sampling...")

            for _ in range(len(df) // self._batch_size):
                if self._shutdown_event.is_set():
                    break

                self._sleep_if_memory_high()
                self._sleep_if_buffer_full()

                self._buffer.put(self._sample_from_df(df=df))

            del df
            local_logger.debug("Cleaned up for sampling, waiting for the next iteration...")

    def _sleep_if_memory_high(self) -> None:
        """Sleep if the memory usage is high"""

        while psutil.virtual_memory().percent > 90.0:
            local_logger.debug("Memory usage is high. Waiting for the memory to be freed...")
            sleep(self._hold_on_time)

    def _sleep_if_buffer_full(self) -> None:
        """Sleep if the buffer queue is full"""

        while self._buffer.full():
            local_logger.debug("Buffer queue is full. Waiting for batches to be loaded...")
            sleep(self._hold_on_time)

    def generate_tiny_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate a tiny dataset from the parquet file

        Args:
            n_samples (int): The number of samples

        Returns:
            df (pd.DataFrame): The tiny dataset
        """

        n_samples_per_group = n_samples // len(self._groups)
        dfs: list[pd.DataFrame] = []

        for g in self._groups:
            dfs.append(self._parquet.read_row_group(g).to_pandas().sample(n_samples_per_group))

        # Concatenate all DataFrames
        df = pd.concat(dfs, ignore_index=True)

        remaining_samples = n_samples - len(df)
        if remaining_samples > 0:
            additional_df = (
                self._parquet.read_row_groups(row_groups=self._get_rows_group(1)).to_pandas().sample(remaining_samples)
            )
            df = pd.concat([df, additional_df], ignore_index=True)

        return df

    def get_batch(self, size: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Get a batch of data
        NOTE: This method is used for testing purposes only.

        Args:
            size (int): The batch size

        Returns:
            x (np.ndarray | torch.Tensor): The input data
                - np.ndarray: if is_to_tensor is False
                - torch.Tensor: if is_to_tensor is True
            y (np.ndarray | torch.Tensor): The output data
                - np.ndarray: if is_to_tensor is False
                - torch.Tensor: if is_to_tensor is True
        """

        df = self._parquet.read_row_groups(
            row_groups=self._get_rows_group(self._n_group_per_sampling),
        ).to_pandas()

        return self._sample_from_df(df=df, batch_size=size)

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a torch DataLoader object
        NOTE: This dataloader uses BatchSampler to return a batch of data.

        Args:
            **kwargs: Additional arguments for DataLoader

        Returns:
            dataloader (torch.utils.data.DataLoader): The DataLoader object
        """

        return torch.utils.data.DataLoader(
            self,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(self),
                batch_size=1,
                drop_last=False,
            ),
            **kwargs,
        )
