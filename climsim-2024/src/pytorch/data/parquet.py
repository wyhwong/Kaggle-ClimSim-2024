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
        batch_size: int = src.env.BATCH_SIZE,
        buffer_size: int = src.env.BUFFER_SIZE,
        hold_on_time: float = 0.2,  # 0.2 seconds
        is_to_tensor: bool = True,
        is_normalize: bool = src.env.IS_NORMALIZED,
        is_standardize: bool = src.env.IS_STANDARDIZED,
        n_group_per_sampling: Optional[int] = src.env.N_GROUP_PER_SAMPLING,
        groups: Optional[list[int]] = None,
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
            is_to_tensor (bool): Convert the data to GPU tensors or not
            is_normalize (bool): Normalize the data or not
            is_standardize (bool): Standardize the data or not
            n_group_per_sampling (int): The number of groups to be sampled per iteration
            groups (list[int]): The groups to be used for sampling

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
        # If n_group_per_sampling is not provided, use all the groups
        self._n_group_per_sampling = n_group_per_sampling or self._parquet.num_row_groups
        self._buffer_size = buffer_size
        self._hold_on_time = hold_on_time
        self._is_to_tensor = is_to_tensor
        self._is_normalize = is_normalize
        self._is_standardize = is_standardize

        self._n_samples = sum([self._parquet.metadata.row_group(g).num_rows for g in self._groups])
        self._max_idx = self.__len__() - 1

        self._x_min = self._x_norm_scaling = self._y_min = self._y_norm_scaling = np.array([])
        self._init_normalization_scaling()
        self._x_std_mean = self._x_std_scaling = self._y_std_mean = self._y_std_scaling = np.array([])
        self._init_standardization_scaling()

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

        if self._is_to_tensor:
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

    def _init_normalization_scaling(self) -> None:
        """Get the scaling values for normalization"""

        self._x_min = self._x_stats.loc["min"].values
        self._x_norm_scaling = self._x_stats.loc["max"].values - self._x_min
        self._y_min = self._y_stats.loc["min"].values
        self._y_norm_scaling = self._y_stats.loc["max"].values - self._y_min

        # Replace 0 with 1 to avoid division by zero
        # If max is min, then the normalized value is always 0
        self._x_norm_scaling[self._x_norm_scaling == 0] = 1.0
        self._y_norm_scaling[self._y_norm_scaling == 0] = 1.0

    def _init_standardization_scaling(self) -> None:
        """Get the scaling values for standardization"""

        if not self._is_normalize:
            self._x_std_mean = self._x_stats.loc["mean"].values
            self._x_std_scaling = self._x_stats.loc["std"].values
            self._y_std_mean = self._y_stats.loc["mean"].values
            self._y_std_scaling = self._y_stats.loc["std"].values

        else:
            self._x_std_mean = self._x_stats.loc["norm_mean"].values
            self._x_std_scaling = self._x_stats.loc["norm_std"].values
            self._y_std_mean = self._y_stats.loc["norm_mean"].values
            self._y_std_scaling = self._y_stats.loc["norm_std"].values

        # Replace 0 with 1 to avoid division by zero
        # If std is 0, then the standardized value is always 0
        self._x_std_scaling[self._x_std_scaling == 0] = 1.0
        self._y_std_scaling[self._y_std_scaling == 0] = 1.0

    def _get_rows_group(self, size: int) -> np.ndarray:
        """Return a random group"""

        return np.random.choice(self._groups, size=size, replace=False)

    def _sample_from_df(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Sample from a DataFrame"""

        samples = df.sample(self._batch_size)
        x, y = samples[self._input_cols].values, samples[self._target_cols].values

        if self._is_normalize:
            return self._normalize_batch(x, y)

        if self._is_standardize:
            return self._standardize_batch(x, y)

        return x, y

    def _normalize_batch(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize the data"""

        x = (x - self._x_min) / self._x_norm_scaling
        y = (y - self._y_min) / self._y_norm_scaling
        return x, y

    def _standardize_batch(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardize the data"""

        x = (x - self._x_std_mean) / self._x_std_scaling
        y = (y - self._y_std_mean) / self._y_std_scaling
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
            local_logger.debug("Loaded the row groups for sampling...")

            for _ in range(len(df) // self._batch_size):
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

        if self._is_to_tensor:
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
        """Get a batch of data
        NOTE: This method is used for testing purposes only.
        """

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

        if self._is_normalize:
            x = (x - self._x_min) / self._x_norm_scaling

        if self._is_standardize:
            x = (x - self._x_std_mean) / self._x_std_scaling

        df[self._input_cols] = x
        return df

    def postprocess_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Denormalize the data"""

        y = df[self._target_cols].values

        # NOTE:
        # We did normalization first and then standardization
        # So here we first unstardardize and then denormalize
        if self._is_standardize:
            y[:, self._y_std_scaling == 1] = 0.0
            y = y * self._y_std_scaling + self._y_std_mean

        if self._is_normalize:
            y[:, self._y_norm_scaling == 1] = 0.0
            y = y * self._y_norm_scaling + self._y_min

        df[self._target_cols] = y
        return df
