import numpy as np
import pandas as pd
import torch

import src.env
import src.logger
import src.pytorch.dataset.base as base


local_logger = src.logger.get_logger(__name__)


class FileDataset(base.DatasetBase):
    """PyTorch dataset for parquet files"""

    def __init__(
        self,
        source: str,
        x_stats: str,
        y_stats: str,
        device: str = "cuda",
        is_to_tensor: bool = True,
        is_normalize: bool = False,
        is_standardize: bool = True,
    ) -> None:
        """TinyDataset constructor"""

        super().__init__(source, x_stats, y_stats, device, is_to_tensor, is_normalize, is_standardize)

        data = self.read_source()

        self.x = self.preprocess_features(data[self.input_cols].values)
        self.y = self.preprocess_targets(data[self.output_cols].values)

    def read_source(self) -> pd.DataFrame:
        """Read the source file"""

        if ".parquet" in self._source:
            return pd.read_parquet(self._source)
        elif ".csv" in self._source:
            return pd.read_csv(self._source)
        elif ".arrow" in self._source or ".feather" in self._source:
            return pd.read_feather(self._source)
        raise ValueError("Unknown file format")

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        """Return the idx-th sample from the dataset"""

        return self.x[idx], self.y[idx]

    def get_batch(self, size: int) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data
        NOTE: This method is used for testing purposes only.
        """

        idx = np.random.randint(0, len(self), size)
        return self.x[idx], self.y[idx]

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a torch DataLoader object"""

        return torch.utils.data.DataLoader(self, **kwargs)
