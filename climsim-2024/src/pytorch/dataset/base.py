from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


class DatasetBase(torch.utils.data.Dataset, ABC):
    """Base class for PyTorch datasets"""

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
        """DatasetBase constructor

        Args:
            source (str): The source of the dataset
            x_stats (str): The path to the x statistics
            y_stats (str): The path to the y statistics
            device (str): The device to use
            is_to_tensor (bool): Whether to convert to tensor
            is_normalize (bool): Whether to normalize the data
            is_standardize (bool): Whether to standardize the data
        """

        super().__init__()

        self._source = source
        self._device = device

        self.x_stats = pd.read_parquet(x_stats)
        self.y_stats = pd.read_parquet(y_stats)
        self.input_cols = self.x_stats.columns.tolist()
        self.output_cols = self.y_stats.columns.tolist()

        self._is_to_tensor = is_to_tensor
        self._is_normalize = is_normalize
        self._is_standardize = is_standardize

        self._x_min = self._x_norm_scaling = self._y_min = self._y_norm_scaling = np.array([])
        self._init_normalization_scaling()
        self._x_std_mean = self._x_std_scaling = self._y_std_mean = self._y_std_scaling = np.array([])
        self._init_standardization_scaling()

    def _init_normalization_scaling(self) -> None:
        """Get the scaling values for normalization"""

        self._x_min = self.x_stats.loc["min"].values
        self._x_norm_scaling = self.x_stats.loc["max"].values - self._x_min
        self._y_min = self.y_stats.loc["min"].values
        self._y_norm_scaling = self.y_stats.loc["max"].values - self._y_min

        # Replace 0 with 1 to avoid division by zero
        # If max is min, then the normalized value is always 0
        self._x_norm_scaling[self._x_norm_scaling == 0] = 1.0
        self._y_norm_scaling[self._y_norm_scaling == 0] = 1.0

    def _init_standardization_scaling(self) -> None:
        """Get the scaling values for standardization"""

        if not self._is_normalize:
            self._x_std_mean = self.x_stats.loc["mean"].values
            self._x_std_scaling = self.x_stats.loc["std"].values
            self._y_std_mean = self.y_stats.loc["mean"].values
            self._y_std_scaling = self.y_stats.loc["std"].values

        else:
            self._x_std_mean = self.x_stats.loc["norm_mean"].values
            self._x_std_scaling = self.x_stats.loc["norm_std"].values
            self._y_std_mean = self.y_stats.loc["norm_mean"].values
            self._y_std_scaling = self.y_stats.loc["norm_std"].values

        # Replace 0 with 1 to avoid division by zero
        # If std is 0, then the standardized value is always 0
        self._x_std_scaling[self._x_std_scaling == 0] = 1.0
        self._y_std_scaling[self._y_std_scaling == 0] = 1.0

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Normalize the input"""

        x = (x - self._x_min) / self._x_norm_scaling
        return x

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize the output"""

        y = (y - self._y_min) / self._y_norm_scaling
        return y

    def _standardize_x(self, x: np.ndarray) -> np.ndarray:
        """Standardize the input"""

        x = (x - self._x_std_mean) / self._x_std_scaling
        return x

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        """Standardize the output"""

        y = (y - self._y_std_mean) / self._y_std_scaling
        return y

    def _to_tensor(self, inputs: np.ndarray) -> torch.Tensor:
        """Convert the input and output to PyTorch tensors"""

        return torch.Tensor(inputs).to(self._device)

    def preprocess_features(self, x: np.ndarray) -> np.ndarray | torch.Tensor:
        """Preprocess the input"""

        if self._is_normalize:
            x = self._normalize_x(x)

        if self._is_standardize:
            x = self._standardize_x(x)

        if self._is_to_tensor:
            return self._to_tensor(x)
        return x

    def preprocess_targets(self, y: np.ndarray) -> np.ndarray | torch.Tensor:
        """Preprocess the output"""

        if self._is_normalize:
            y = self._normalize_y(y)

        if self._is_standardize:
            y = self._standardize_y(y)

        if self._is_to_tensor:
            return self._to_tensor(y)
        return y

    def postprocess_targets(self, y: np.ndarray) -> np.ndarray:
        """Postprocess the output"""

        if self._is_standardize:
            y[:, self._y_std_scaling == 1] = 0.0
            y = y * self._y_std_scaling + self._y_std_mean

        if self._is_normalize:
            y[:, self._y_norm_scaling == 1] = 0.0
            y = y * self._y_norm_scaling + self._y_min

        return y

    @abstractmethod
    def get_batch(self, size: int):
        """Return a batch of data"""

    @abstractmethod
    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Return a DataLoader object"""
