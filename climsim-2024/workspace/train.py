import os

import torch
from lightning import LightningDataModule
from torch.utils.data import random_split

from src.pytorch.dataset import MemoryParquetDataset, compute_dataset_statistics
from src.pytorch.models import MLP
from src.pytorch.utils import get_default_trainer


torch.set_float32_matmul_precision("highest")

# Training parameters
FULL_DATASET_PATH = "/home/data/train.parquet"
TRAINSET_PATH = "/home/data/train_tiny_1000000.parquet"
X_STATS_PATH = "/home/data/x_stats.parquet"
Y_STATS_PATH = "/home/data/y_stats.parquet"
MODEL_NAME = "climsim_tiny_model"
TRAINING_SAMPLE_FRAC = 0.8
BATCH_SIZE = 1024
N_EPOCHS = 50


# Compute and save dataset statistics if not already done
if not (os.path.exists(X_STATS_PATH) and os.path.exists(Y_STATS_PATH)):
    df_x_stats, df_y_stats = compute_dataset_statistics(FULL_DATASET_PATH)
    df_x_stats.to_parquet(X_STATS_PATH)
    df_y_stats.to_parquet(Y_STATS_PATH)


class ClimSimDataModule(LightningDataModule):
    """Data module for the ClimSim dataset."""

    def __init__(self) -> None:
        """Initialize the data module."""

        super().__init__()

        dataset = MemoryParquetDataset(
            source=TRAINSET_PATH,
            x_stats=X_STATS_PATH,
            y_stats=Y_STATS_PATH,
        )
        train_size = int(TRAINING_SAMPLE_FRAC * len(dataset))
        val_size = len(dataset) - train_size
        self.train, self.val = random_split(
            dataset=dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self):
        """Return the training dataloader."""

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""

        return torch.utils.data.DataLoader(
            self.val,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )


def train_model():
    """Train the model."""

    datamodule = ClimSimDataModule()
    model = MLP()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_iters = int(N_EPOCHS * len(datamodule.train) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=total_iters)
    model.replace_optimizers(optimizers=[optimizer], schedulers=[scheduler])

    trainer = get_default_trainer(
        deterministic=False,
        model_name=MODEL_NAME,
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_model()
