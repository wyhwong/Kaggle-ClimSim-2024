import os

os.environ["RANDOM_SEED"] = "2024"
os.environ["BATCH_SIZE"] = "1024"
os.environ["N_EPOCHS"] = "50"
os.environ["MAXIMUM_TRAINING_TIME_IN_HOUR"] = "72"

import datetime

import torch
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import random_split

import src.env
from src.pytorch.data.parquet_tiny import TinyDataset
from src.pytorch.data.statistics import compute_dataset_statistics
from src.pytorch.models.mlp import MLP
from src.pytorch.models.utils import get_default_trainer


torch.set_float32_matmul_precision("highest")

# Training parameters
TRAINSET_DATA_PATH = "/home/data/train_tiny_1000000.parquet"
X_STATS_PATH = "/home/data/x_stats.parquet"
Y_STATS_PATH = "/home/data/y_stats.parquet"
MODEL_NAME = "climsim_tiny_model"
TRAINING_SAMPLE_FRAC = 0.8


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

        data = pd.read_parquet(data_path)
        dataset = TinyDataset(
            data=data,
            x_stats=X_STATS_PATH,
            y_stats=Y_STATS_PATH,
        )
        train_size = int(TRAINING_SAMPLE_FRAC * len(data))
        val_size = len(data) - train_size
        self.train, self.val = random_split(
            dataset=dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self):
        """Return the training dataloader."""

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=src.env.BATCH_SIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""

        return torch.utils.data.DataLoader(
            self.val,
            batch_size=src.env.BATCH_SIZE,
            shuffle=False,
        )


def train_model():
    """Train the model."""

    datamodule = ClimSimDataModule(TRAINSET_DATA_PATH)
    model = MLP(steps_per_epoch=len(datamodule.train), layers_hidden=[556, 1024, 512, 368])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_iters = int(src.env.N_EPOCHS * len(datamodule.train) / src.env.BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=total_iters)
    model.replace_optimizers(optimizers=[optimizer], schedulers=[scheduler])
    trainer = get_default_trainer(
        deterministic=False,
        model_name=MODEL_NAME,
        max_time=datetime.timedelta(days=3),
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_model()
