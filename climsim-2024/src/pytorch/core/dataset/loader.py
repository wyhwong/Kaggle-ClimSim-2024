from typing import Union

import numpy as np
import polars as pl
import torch

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A Dataset class for the torch dataloader"""

    def __init__(
        self,
        lazyframe: pl.LazyFrame,
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
    ) -> None:
        """
        Initialize the Dataset

        Args:
            lazyframe (pl.LazyFrame): The dataset to be used
            input_cols (list[str]): The input columns
            target_cols (list[str]): The target columns
            batch_size (int): The batch size to be used

        Returns:
            None
        """

        self._lf = lazyframe
        self._input_cols = input_cols
        self._target_cols = target_cols
        self._batch_size = batch_size

        self._n_updates = 0
        self._X: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])

    def __len__(self) -> int:
        """Return the length of the dataset"""

        return self._lf.select(pl.len()).collect().item()

    def __getitem__(self, idx: int):
        """Return the data at the given index"""

        idx %= self._batch_size
        if idx == 0:
            self._update_batch()

        return self._X[idx], self._y[idx]

    def _update_batch(self) -> None:
        """Get a batch of the dataset"""

        samples = (
            self._lf.with_columns(pl.all().shuffle(seed=src.env.RANDOM_SEED + self._n_updates))
            .head(self._batch_size)
            .collect()
        )
        self._X, self._y = samples[self._input_cols].to_numpy(), samples[self._target_cols].to_numpy()
        self._n_updates += 1


class DatasetHandler:
    """A class to handle the dataset"""

    def __init__(
        self,
        dataset: Union[pl.DataFrame, pl.LazyFrame],
        input_cols: list[str],
        target_cols: list[str],
        batch_size: int,
        shuffle: bool = False,
        train_fraction: float = 0.8,
    ) -> None:
        """
        Initialize the DatasetHandler

        Args:
            dataset (Union[pl.DataFrame, pl.LazyFrame]): The dataset to be used
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

        self._trainset, self._valset = self._split_dataset()

    def _sample(self, s: pl.Series) -> pl.Series:
        """Sampling function for train-val split"""

        # Here we need to reset the random seed for reproducibility
        # Or else, trainset will vary during training
        np.random.seed(src.env.RANDOM_SEED)
        return pl.Series(
            values=np.random.binomial(1, self._train_fraction, s.len()),
            dtype=pl.Boolean,
        )

    def _split_dataset(self) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Split the dataset into training and validation sets

        Returns:
            tuple[pl.LazyFrame, pl.LazyFrame]: The training and validation datasets
        """

        if isinstance(self.dataset, pl.DataFrame):
            lf = self.dataset.lazy()
        else:
            lf = self.dataset

        if self._shuffle:
            lf = lf.with_columns(pl.all().shuffle(seed=src.env.RANDOM_SEED))

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
