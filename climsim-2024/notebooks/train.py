import datetime
import random
from typing import Optional

import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule

from src.utils import check_and_create_dir
from src.pytorch.models.utils import get_default_trainer
from src.pytorch.models.mlp import MLP
from src.pytorch.models.fastkan import FastKAN
from src.pytorch.data.parquet import Dataset
from src.pytorch.loss.r2 import r2_residual_multivariate
from src.schemas.climsim import INPUT_COLUMNS, OUTPUT_COLUMNS


torch.set_float32_matmul_precision("high")

# Training parameters
TRAINSET_DATA_PATH = "/home/data/train.parquet"
OUTPUT_DIR = "./results"
check_and_create_dir(OUTPUT_DIR)
MODEL_NAME = "climsim_best_model"
BATCH_SIZE = 3072
N_EPOCHS = 600
TRAINING_SAMPLE_FRAC = 0.7
# BUFFER_SIZE: number of batches being preloaded in memory
TRAINING_BUFFER_SIZE = 100
VALIDATION_BUFFER_SIZE = 100
# N_GROUP_PER_SAMPLING: number of groups being sampled in each iteration
TRAINING_N_GROUP_PER_SAMPLING = 3
VALIDATION_N_GROUP_PER_SAMPLING = 1
# N_BATCH_PER_SAMPLING: number of batches being sampled in each iteration
TRAINING_N_BATCH_PER_SAMPLING = 100
VALIDATION_N_BATCH_PER_SAMPLING = 100
IS_NORMALIZED = True
IS_UNLEARNABLE_DROPPED = False


class ClimSimDataModule(LightningDataModule):
    """Data module for the ClimSim dataset."""

    def __init__(self, data_path: str, batch_size: int) -> None:
        """Initialize the data module."""

        super().__init__()

        self._data_path = data_path
        self._batch_size = batch_size

    def setup(self, stage: Optional[str] = None):

        # Use full dataset for training
        parquet = pq.ParquetFile(self._data_path, memory_map=True, buffer_size=10)
        all_groups = list(range(0, parquet.num_row_groups))
        train_groups = random.sample(all_groups, int(TRAINING_SAMPLE_FRAC * len(all_groups)))
        val_groups = list(set(all_groups) - set(train_groups))

        self.train = Dataset(
            source=self._data_path,
            input_cols=INPUT_COLUMNS,
            target_cols=OUTPUT_COLUMNS,
            batch_size=self._batch_size,
            buffer_size=TRAINING_BUFFER_SIZE,
            groups=train_groups,
            n_group_per_sampling=TRAINING_N_GROUP_PER_SAMPLING,
            n_batch_per_sampling=TRAINING_N_BATCH_PER_SAMPLING,
            to_tensor=True,
            normalize=IS_NORMALIZED,
            drop_unlearnable=IS_UNLEARNABLE_DROPPED,
        )
        self.val = Dataset(
            source=self._data_path,
            input_cols=INPUT_COLUMNS,
            target_cols=OUTPUT_COLUMNS,
            batch_size=self._batch_size,
            buffer_size=VALIDATION_BUFFER_SIZE,
            groups=val_groups,
            n_group_per_sampling=VALIDATION_N_GROUP_PER_SAMPLING,
            n_batch_per_sampling=VALIDATION_N_BATCH_PER_SAMPLING,
            to_tensor=True,
            normalize=IS_NORMALIZED,
            drop_unlearnable=IS_UNLEARNABLE_DROPPED,
        )

    def train_dataloader(self):
        """Return the training dataloader."""

        return self.train.to_dataloader()

    def val_dataloader(self):
        """Return the validation dataloader."""

        return self.val.to_dataloader()

    def teardown(self, stage: str) -> None:
        """Clean up the data module."""

        self.train.clean_up()
        self.val.clean_up()


def train_model():
    """Train the model."""

    model = MLP(loss_fn=r2_residual_multivariate)
    datamodule = ClimSimDataModule(TRAINSET_DATA_PATH, BATCH_SIZE)
    trainer = get_default_trainer(
        model_name=MODEL_NAME,
        max_epochs=N_EPOCHS,
        max_time=datetime.timedelta(days=3),
        check_val_every_n_epoch=3,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_model()
