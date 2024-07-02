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


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        source: str,
        x_stats: str,
        y_stats: str,
        batch_size: int = 3072,
        buffer_size: int = 10,  # 10 batches
        hold_on_time: float = 0.2,  # 0.2 seconds
        groups: Optional[list[int]] = None,
        n_batch_per_sampling: Optional[int] = None,
        n_group_per_sampling: Optional[int] = None,
        to_tensor: bool = True,
        normalize: bool = False,
        standardize: bool = False,
        drop_unlearnable: bool = False,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            source (str): The path to the source file
            x_stats (str): The statistics of the input columns
            y_stats (str): The statistics of the target columns
            batch_size (int): The batch size to be used
            buffer_size (int): The buffer size to be used
            hold_on_time (float): The time to wait before checking the buffer again (if the buffer is full)
            groups (list[int]): The groups to be used for sampling
            n_batch_per_sampling (int): The number of batches to be sampled per group
            n_group_per_sampling (int): The number of groups to be sampled per iteration
            to_tensor (bool): Convert the data to GPU tensors
            normalize (bool): Normalize the data
            standardize (bool): Standardize the data
            drop_unlearnable (bool): Drop the unlearnable data

        Returns:
            None
        """

        self._parquet = pq.ParquetFile(source, memory_map=True, buffer_size=10)
        self._x_stats = pd.read_parquet(x_stats)
        self._y_stats = pd.read_parquet(y_stats)
        self._input_cols = self._x_stats.columns.tolist()
        self._target_cols = self._y_stats.columns.tolist()
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
        self._standardize = standardize
        self._drop_unlearnable = drop_unlearnable

        self._n_samples = sum([self._parquet.metadata.row_group(g).num_rows for g in self._groups])
        self._max_idx = self.__len__() - 1

        self._x_min = self._x_scaling = self._y_min = self._y_scaling = np.array([])
        self._init_norm_scaling()
        self._norm_x_mean = self._norm_x_std = self._norm_y_mean = self._norm_y_std = np.array([])
        self._init_standard_scaling()

        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._sampling_thread = threading.Thread(target=lambda: None, daemon=True)
        self._np_buffer: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=self._buffer_size)
        self._tensor_buffer: Queue[tuple[torch.Tensor, torch.Tensor]] = Queue(maxsize=self._buffer_size)

    def __len__(self) -> int:
        """Return the length of the dataset (unit in batch)"""

        return int(self._n_samples / self._batch_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index from the buffer"""

        if not self._sampling_thread.is_alive():
            self.start_sampling_worker()

        if idx == self._max_idx:
            self.shutdown_sampling_worker()
            self.start_sampling_worker()

        if self._np_buffer.empty() and self._tensor_buffer.empty():
            local_logger.debug("Buffer queue is empty. Waiting for batches to be loaded...")

        if self._to_tensor:
            return self._tensor_buffer.get()

        return self._np_buffer.get()

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

        self._np_buffer.queue.clear()
        self._tensor_buffer.queue.clear()

    def get_cols(self) -> tuple[list[str], list[str]]:
        """Get the input and target columns"""

        return self._input_cols, self._target_cols

    def _init_norm_scaling(self) -> None:
        """Get the scaling values for normalization"""

        self._x_min = self._x_stats.loc["min"].values
        self._x_scaling = self._x_stats.loc["max"].values - self._x_min
        if self._drop_unlearnable:
            unlearnable_idx = np.nonzero(self._x_scaling == 0)[0]
            unlearnable_cols = [self._input_cols[i] for i in unlearnable_idx]
            self._input_cols = [col for col in self._input_cols if col not in unlearnable_cols]
            self._x_min = self._x_stats.loc["min"].values
            self._x_scaling = self._x_stats.loc["max"].values - self._x_min
            local_logger.warning(
                "Dropped unlearnable columns: %s. Please check whether this is expected.", unlearnable_cols
            )
            local_logger.info("Updated input columns: %s.", self._input_cols)

        else:
            self._x_scaling[self._x_scaling == 0] = 1.0

        self._y_min = self._y_stats.loc["min"].values
        self._y_scaling = self._y_stats.loc["max"].values - self._y_min
        if any(self._y_scaling == 0):
            unlearnable_idx = np.nonzero(self._y_scaling == 0)[0]
            local_logger.warning(
                "Unlearnable columns in the target: %s. Please check whether this is expected.",
                [self._target_cols[i] for i in unlearnable_idx],
            )
            self._y_scaling[self._y_scaling == 0] = 1.0

    def _init_standard_scaling(self) -> None:
        """Get the scaling values for standardization"""

        if not self._normalize:
            self._norm_x_mean = self._x_stats.loc["mean"].values
            self._norm_x_std = self._x_stats.loc["std"].values
            self._norm_y_mean = self._y_stats.loc["mean"].values
            self._norm_y_std = self._y_stats.loc["std"].values

        else:
            self._norm_x_mean = self._x_stats.loc["norm_mean"].values
            self._norm_x_std = self._x_stats.loc["norm_std"].values
            self._norm_y_mean = self._y_stats.loc["norm_mean"].values
            self._norm_y_std = self._y_stats.loc["norm_std"].values

    def _get_rows_group(self, size: int) -> np.ndarray:
        """Return a random group"""

        return np.random.choice(self._groups, size=size, replace=False)

    def _sample_from_df(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Sample from a DataFrame"""

        samples = df.sample(self._batch_size).reset_index(drop=True)
        x, y = samples[self._input_cols].values, samples[self._target_cols].values

        if self._normalize:
            return self._normalize_batch(x, y)

        if self._standardize:
            return self._standardize_batch(x, y)

        return x, y

    def _normalize_batch(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize the data"""

        x = (x - self._x_min) / self._x_scaling
        y = (y - self._y_min) / self._y_scaling
        return x, y

    def _standardize_batch(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardize the data"""

        x = (x - self._norm_x_mean) / self._norm_x_std
        y = (y - self._norm_y_mean) / self._norm_y_std
        return x, y

    def _put_batch_to_tensor_buffer(self, x: np.ndarray, y: np.ndarray) -> None:
        """Convert the data to GPU tensors"""

        _x = torch.Tensor(x).to(src.env.DEVICE)
        _y = torch.Tensor(y).to(src.env.DEVICE)
        self._tensor_buffer.put((_x, _y))

    def _sampling_worker_fn(self) -> None:
        """Load batches in the background and populate the buffer queue"""

        while True:
            if self._shutdown_event.is_set():
                local_logger.debug("Event has been set. Shutting down the worker...")
                return

            df = self._parquet.read_row_groups(
                row_groups=self._get_rows_group(self._n_group_per_sampling),
            ).to_pandas()

            for _ in range(self._n_batch_per_sampling):
                if self._shutdown_event.is_set():
                    break

                self._sleep_if_memory_high()
                self._sleep_if_buffer_full()

                x, y = self._sample_from_df(df=df)
                self._put_batch_to_tensor_buffer(x, y)

            del df
            local_logger.debug("Cleaned up for sampling, waiting for the next iteration...")

    def _put_samples_to_buffer(self, x: np.ndarray, y: np.ndarray) -> None:
        """Put the samples to the buffer"""

        if self._to_tensor:
            self._put_batch_to_tensor_buffer(x, y)
        else:
            self._np_buffer.put((x, y))

    def _sleep_if_memory_high(self) -> None:
        """Sleep if the memory usage is high"""

        while psutil.virtual_memory().percent > 90.0:
            local_logger.debug("Memory usage is high. Waiting for the memory to be freed...")
            sleep(self._hold_on_time)

    def _sleep_if_buffer_full(self) -> None:
        """Sleep if the buffer queue is full"""

        while self._np_buffer.full() or self._tensor_buffer.full():
            local_logger.debug("Buffer queue is full. Waiting for batches to be loaded...")
            sleep(self._hold_on_time)

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

    def get_batch(self, is_tensor: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from the buffer"""

        df = self._parquet.read_row_groups(
            row_groups=self._get_rows_group(self._n_group_per_sampling),
        ).to_pandas()

        x, y = self._sample_from_df(df=df)

        if is_tensor:
            return torch.Tensor(x).to(src.env.DEVICE), torch.Tensor(y).to(src.env.DEVICE)
        return x, y

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data"""

        x = df[self._input_cols].values

        if self._normalize:
            x = (x - self._x_min) / self._x_scaling

        if self._standardize:
            x = (x - self._norm_x_mean) / self._norm_x_std

        df[self._input_cols] = x
        return df

    def postprocess_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Denormalize the data"""

        y = df[self._target_cols].values

        if self._normalize:
            y = y * self._y_scaling + self._y_min

        if self._standardize:
            y = y * self._norm_y_std + self._norm_y_mean

        df[self._target_cols] = y
        return df
