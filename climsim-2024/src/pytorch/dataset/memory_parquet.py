import numpy as np
import pandas as pd
import torch

import src.env
import src.logger
import src.pytorch.dataset.base as base


local_logger = src.logger.get_logger(__name__)


class MemoryParquetDataset(base.DatasetBase):
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
        """MemoryParquetDataset constructor

        Args:
            source (str): The path to the dataset
            x_stats (str): The path to the x statistics (parquet file)
            y_stats (str): The path to the y statistics (parquet file)
            device (str): The device to use
            is_to_tensor (bool): Whether to convert to tensor
            is_normalize (bool): Whether to normalize the data
            is_standardize (bool): Whether to standardize the data

        Methods (excluding inherited methods):
            read_source: Read the source file
            __len__: Return the number of samples in the dataset
            __getitem__: Return the idx-th sample from the dataset
            get_batch: Get a batch of data
            to_dataloader: Return a torch DataLoader object
        """

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

    def get_batch(self, size: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Get a batch of data
        NOTE: This method is used for testing purposes only.

        Args:
            size (int): The batch size

        Returns:
            x (np.ndarray | torch.Tensor): The input data
                - np.ndarray: if is_to_tensor is False
                - torch.Tensor: if is_to_tensor is True
            y (np.ndarray | torch.Tensor): The output data
                - np.ndarray: if is_to_tensor is False
                - torch.Tensor: if is_to_tensor is True
        """

        idx = np.random.randint(0, len(self), size)
        return self.x[idx], self.y[idx]

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a torch DataLoader object

        Args:
            **kwargs: Additional arguments for DataLoader

        Returns:
            dataloader (torch.utils.data.DataLoader): The DataLoader object
        """

        return torch.utils.data.DataLoader(self, **kwargs)
