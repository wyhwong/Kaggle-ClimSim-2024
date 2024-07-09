import pandas as pd
import torch

import src.logger


local_logger = src.logger.get_logger(__name__)


class TinyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for parquet files"""

    def __init__(self, data: pd.DataFrame, x_stats: str, y_stats: str) -> None:
        """TinyDataset constructor"""

        super().__init__()

        self._x_stats = pd.read_parquet(x_stats)
        self._y_stats = pd.read_parquet(y_stats)
        self._input_cols = self._x_stats.columns.tolist()
        self._target_cols = self._y_stats.columns.tolist()

        self.x = data[self._input_cols]
        self.x = self.x.to_numpy()
        self.x = torch.from_numpy(self.x)

        self.y = data[self._target_cols]
        self.y = self.y.to_numpy()
        self.y = torch.from_numpy(self.y)

        self.x_mean = self._x_stats.loc["mean"].to_numpy()
        self.x_std = self._x_stats.loc["std"].to_numpy()
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = self._y_stats.loc["mean"].to_numpy()
        self.y_std = self._y_stats.loc["std"].to_numpy()
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
