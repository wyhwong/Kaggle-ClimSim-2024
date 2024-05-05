import numpy as np
import pyarrow.parquet as pq
import torch

import src.env
import src.logger
import src.schemas.math


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        fparquet: pq.ParquetFile,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
        n_samples: int,
        groups: list[int],
        train_fraction: float = 0.8,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            fparquet (pq.ParquetFile): The dataset to be used
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used
            n_samples (int): The number of samples in the dataset
            train_fraction (float): The fraction of data to be used for training

        Returns:
            None
        """

        self._lf = fparquet
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._batch_size = batch_size
        self._n_samples = n_samples
        self._train_fraction = train_fraction

        self._n_updates = 0
        self._X: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])
        self._group_fn = lambda: np.random.choice(groups)

    def __len__(self) -> int:
        """Return the length of the dataset"""

        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index"""

        idx %= self._batch_size
        if idx == 0:
            self._update_batch()

        return (self._X[idx], self._y[idx])

    def _update_batch(self) -> None:
        """Get a batch of the dataset"""

        df = self._lf.read_row_group(self._group_fn()).to_pandas()
        samples = df.sample(self._batch_size)
        self._X, self._y = samples[self._input_cols].values, samples[self._target_cols].values
        self._n_updates += 1
