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
        """BufferedParquetDataset constructor"""

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
            self.start_sampling_worker()

        if idx + 1 == self.__len__():
            self.shutdown_sampling_worker()
            self.start_sampling_worker()

        if self._buffer.empty():
            local_logger.debug("Buffer queue is empty. Waiting for batches to be loaded...")

        return self._buffer.get()

    def start_sampling_worker(self) -> None:
        """Start the worker threads"""

        self._lock.acquire()
        self._shutdown_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sampling_worker_fn, daemon=True)
        self._sampling_thread.start()
        self._lock.release()
        local_logger.debug("Sampling thread has been started.")

    def shutdown_sampling_worker(self) -> None:
        """Shutdown the threads"""

        self._lock.acquire()
        self._shutdown_event.set()
        self._sampling_thread.join()
        self._lock.release()
        local_logger.debug("Sampling thread has been shut down.")

    def clean_up(self) -> None:
        """Clean up the resources"""

        self._buffer.queue.clear()

    def _get_rows_group(self, size: int) -> np.ndarray:
        """Return a random group"""

        return np.random.choice(self._groups, size=size, replace=False)

    def _sample_from_df(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
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
        """Generate a tiny dataset"""

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
        """

        df = self._parquet.read_row_groups(
            row_groups=self._get_rows_group(self._n_group_per_sampling),
        ).to_pandas()

        return self._sample_from_df(df=df, batch_size=size)

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
