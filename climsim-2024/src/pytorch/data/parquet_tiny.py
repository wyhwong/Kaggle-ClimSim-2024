from typing import Optional

import numpy as np
import pandas as pd
import torch

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


class TinyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for parquet files"""

    def __init__(self, data: pd.DataFrame, x_stats: str, y_stats: str) -> None:
        """TinyDataset constructor"""

        super().__init__()

        self.x_stats = pd.read_parquet(x_stats)
        self.y_stats = pd.read_parquet(y_stats)
        self.input_cols = self.x_stats.columns.tolist()
        self.output_cols = self.y_stats.columns.tolist()

        self.x = data[self.input_cols]
        self.x = self.x.to_numpy()
        self.x = torch.from_numpy(self.x)

        self.y = data[self.output_cols]
        self.y = self.y.to_numpy()
        self.y = torch.from_numpy(self.y)

        self.x_mean = self.x_stats.loc["mean"].to_numpy()
        self.x_std = self.x_stats.loc["std"].to_numpy()
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = self.y_stats.loc["mean"].to_numpy()
        self.y_std = self.y_stats.loc["std"].to_numpy()
        self.y_std[self.y_std == 0] = 1.0

        local_logger.info("Dataset loaded.")

    def __getitem__(self, idx: int) -> tuple:
        """Return the idx-th sample from the dataset"""

        x = self.x[idx]
        y = self.y[idx]

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        return x, y

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.y)

    def get_batch(
        self, size: Optional[int] = None, is_tensor: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of samples"""

        idx = np.random.choice(len(self.y), size=size, replace=False)
        x = self.x[idx]
        y = self.y[idx]

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        if is_tensor:
            return x.to(src.env.DEVICE), y.to(src.env.DEVICE)

        return x.numpy(), y.numpy()

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data"""

        x = df[self.input_cols].values

        x = (x - self.x_mean) / self.x_std

        df[self.input_cols] = x
        return df

    def postprocess_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Denormalize the data"""

        y = df[self.output_cols].values

        y[:, self.y_std == 1] = 0.0
        y = y * self.y_std + self.y_mean

        df[self.output_cols] = y
        return df
