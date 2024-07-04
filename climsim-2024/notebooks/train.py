import os

os.environ["RANDOM_SEED"] = "2024"
os.environ["INITIAL_LR"] = "1e-4"
os.environ["MAXIMUM_LR"] = "1e-3"
os.environ["N_WARMUP_EPOCHS"] = "1"
os.environ["N_DECAY_EPOCHS"] = "22"
os.environ["ALPHA"] = "0.1"
os.environ["BATCH_SIZE"] = "2048"
os.environ["BUFFER_SIZE"] = "100"
os.environ["N_GROUP_PER_SAMPLING"] = "3"
os.environ["N_BATCH_PER_SAMPLING"] = "100"
os.environ["IS_NORMALIZED"] = "False"
os.environ["IS_STANDARDIZED"] = "True"
os.environ["N_EPOCHS"] = "25"
os.environ["MAXIMUM_TRAINING_TIME_IN_HOUR"] = "72"

import datetime
import random
from typing import Optional

import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule

from src.pytorch.data.parquet import Dataset
from src.pytorch.data.statistics import compute_dataset_statistics
from src.pytorch.models.cnn import CNN
from src.pytorch.models.utils import get_default_trainer
from src.utils import check_and_create_dir


torch.set_float32_matmul_precision("highest")

# Training parameters
TRAINSET_DATA_PATH = "/home/data/train.parquet"
X_STATS_PATH = "/home/data/x_stats.parquet"
Y_STATS_PATH = "/home/data/y_stats.parquet"
OUTPUT_DIR = "./results"
check_and_create_dir(OUTPUT_DIR)
MODEL_NAME = "climsim_best_model"
TRAINING_SAMPLE_FRAC = 1.0


# Compute and save dataset statistics if not already done
if not (os.path.exists(X_STATS_PATH) and os.path.exists(Y_STATS_PATH)):
    df_x_stats, df_y_stats = compute_dataset_statistics(TRAINSET_DATA_PATH)
    df_x_stats.to_parquet(X_STATS_PATH)
    df_y_stats.to_parquet(Y_STATS_PATH)


class ClimSimDataModule(LightningDataModule):
    """Data module for the ClimSim dataset."""

    def __init__(self, data_path: str) -> None:
        """Initialize the data module."""

        super().__init__()

        self._parquet = pq.ParquetFile(data_path, memory_map=True, buffer_size=10)
        all_groups = list(range(0, self._parquet.num_row_groups))

        # NOTE: Here we use all groups for training and validation
        train_groups = all_groups
        val_groups = all_groups

        # NOTE: Here we use random sampling for training and validation
        # train_groups = random.sample(all_groups, int(TRAINING_SAMPLE_FRAC * len(all_groups)))
        # val_groups = list(set(all_groups) - set(train_groups))

        self.train = Dataset(
            source=data_path,
            x_stats=X_STATS_PATH,
            y_stats=Y_STATS_PATH,
            groups=train_groups,
        )
        self.val = Dataset(
            source=data_path,
            x_stats=X_STATS_PATH,
            y_stats=Y_STATS_PATH,
            groups=val_groups,
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

    datamodule = ClimSimDataModule(TRAINSET_DATA_PATH)
    model = CNN(steps_per_epoch=len(datamodule.train))
    trainer = get_default_trainer(
        deterministic=False,
        model_name=MODEL_NAME,
        max_time=datetime.timedelta(days=3),
        check_val_every_n_epoch=3,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_model()
