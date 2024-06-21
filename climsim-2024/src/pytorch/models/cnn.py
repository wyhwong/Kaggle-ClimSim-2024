from typing import Callable, Optional

import torch
from torch import nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


# Checked with https://www.kaggle.com/code/abiolatti/keras-baseline-seq2seq
# Same results as keras implementation
def to_sequential_features(x: torch.Tensor) -> torch.Tensor:
    """
    Convert the input tensor to sequential features. (For ClimSim dataset only)

    Args:
        x: The input tensor.

    Returns:
        The input tensor converted to sequential features
    """

    # Reshape and transpose components
    x_seq0 = torch.transpose(x[:, 0 : 60 * 6].reshape(-1, 6, 60), 1, 2)
    x_seq1 = torch.transpose(x[:, 60 * 6 + 16 : 60 * 9 + 16].reshape(-1, 3, 60), 1, 2)
    x_flat = x[:, 60 * 6 : 60 * 6 + 16].reshape(-1, 1, 16)
    x_flat = x_flat.repeat(1, 60, 1)

    # Concatenate along the last dimension and then transpose
    x_combined = torch.cat([x_seq0, x_seq1, x_flat], dim=-1)
    return x_combined.transpose(1, 2)


class CNN(ModelBase):
    """Convolutional neural network model for regression."""

    def __init__(
        self,
        layers_hidden: Optional[nn.Sequential] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a convolutional neural network model for regression.
        """

        super().__init__(loss_fn=loss_fn)

        self._conv_init = nn.Conv1d(in_channels=25, out_channels=64, kernel_size=1, padding="same")
        self._global_avg_pool = nn.AdaptiveAvgPool1d(60)
        self._conv_final = nn.Conv1d(64, 14, kernel_size=1, padding="same")
        self._batch_norm = nn.BatchNorm1d(64)

        self._layers = layers_hidden or nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        x = to_sequential_features(x)
        e0 = self._conv_init(x)
        e = self._layers(e0)
        e = e0 + e + self._global_avg_pool(e)
        e = self._batch_norm(e)
        e = e + self._layers(e)

        p_all = self._conv_final(e)

        p_seq = p_all[:, :6, :]
        p_seq = p_seq.permute(0, 2, 1).reshape(p_seq.shape[0], -1)

        p_flat = p_all[:, 6:14, :]
        p_flat = p_flat.mean(dim=2)

        P = torch.cat([p_seq, p_flat], dim=1)
        return P
