from typing import Optional

import numpy as np
import polars as pl
import torch

import src.env
import src.logger
import src.schemas.math


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        lazyframe: pl.LazyFrame,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
        sample_every_n: Optional[int] = None,
        allowed_offset: Optional[src.schemas.math.Domain] = None,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            lazyframe (pl.LazyFrame): The dataset to be used
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used
            sample_every_n (Optional[int]): The number of samples to skip
            allowed_offset (Optional[src.schemas.math.Domain]): The allowed offset for the target

        Returns:
            None
        """

        self._lf = lazyframe
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._batch_size = batch_size
        self._sample_every_n = sample_every_n
        self._allowed_offset = allowed_offset

        self._n_updates = 0
        self._X: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])

        if self._allowed_offset and self._allowed_offset.right > self._sample_every_n:
            error_msg = "The allowed offset should be less than the sample_every_n"
            local_logger.error(error_msg)
            raise ValueError(error_msg)

        if self._allowed_offset:
            self._offest_fn = lambda: np.random.randint(self._allowed_offset.left, self._allowed_offset.right)
        else:
            self._offest_fn = lambda: np.random.randint(0, self._sample_every_n)

    def __len__(self) -> int:
        """Return the length of the dataset"""

        return self._lf.select(pl.len()).collect().item()

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index"""

        idx %= self._batch_size
        if idx == 0:
            self._update_batch()

        return (self._X[idx], self._y[idx])

    def _update_batch(self) -> None:
        """Get a batch of the dataset"""

        if self._sample_every_n:
            flow = (
                self._lf.gather_every(self._sample_every_n, self._offest_fn())
                .with_columns(pl.all().shuffle(seed=src.env.RANDOM_SEED + self._n_updates))
                .head(self._batch_size)
            )
        else:
            flow = self._lf.with_columns(pl.all().shuffle(seed=src.env.RANDOM_SEED + self._n_updates)).head(
                self._batch_size
            )

        samples = flow.collect()
        self._X, self._y = samples[self._input_cols].to_numpy(), samples[self._target_cols].to_numpy()
        self._n_updates += 1


class DatasetHandler:
    """A class to handle the dataset"""

    def __init__(
        self,
        dataset: pl.LazyFrame,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
        shuffle: bool = False,
        train_fraction: float = 0.8,
    ) -> None:
        """
        Initialize the DatasetHandler

        Args:
            dataset (pl.LazyFrame): The dataset to be used
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used
            shuffle (bool): Whether to shuffle the dataset
            train_fraction (float): The fraction of the dataset to be used for training

        Returns:
            None
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._shuffle = shuffle
        self._train_fraction = train_fraction

        n_samples = self.dataset.select(pl.len()).collect().item()
        self._sampling_series = pl.Series(
            values=np.random.binomial(1, self._train_fraction, n_samples),
            dtype=pl.Boolean,
        )
        self._trainset, self._valset = self._split_dataset()

    def _sample(self, _: pl.Series) -> pl.Series:
        """Sampling function for train-val split"""

        return self._sampling_series

    def _split_dataset(self) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Split the dataset into training and validation sets

        Returns:
            tuple[pl.LazyFrame, pl.LazyFrame]: The training and validation datasets
        """

        if self._shuffle:
            lf = self.dataset.with_columns(pl.all().shuffle(seed=src.env.RANDOM_SEED))
        else:
            lf = self.dataset

        lf = lf.with_columns(pl.first().map_batches(self._sample).alias("_sample"))
        trainset = lf.filter(pl.col("_sample")).drop("_sample")
        valset = lf.filter(pl.col("_sample").not_()).drop("_sample")

        local_logger.info("Trainset size: %d", trainset.select(pl.len()).collect().item())
        local_logger.info("Valset size: %d", valset.select(pl.len()).collect().item())

        return (trainset, valset)

    def get_trainset(self) -> Dataset:
        """Get the training dataset"""

        return Dataset(
            lazyframe=self._trainset,
            batch_size=self.batch_size,
            input_cols=self._input_cols,
            target_cols=self._target_cols,
        )

    def get_valset(self) -> Dataset:
        """Get the validation dataset"""

        return Dataset(
            lazyframe=self._valset,
            batch_size=self.batch_size,
            input_cols=self._input_cols,
            target_cols=self._target_cols,
        )
