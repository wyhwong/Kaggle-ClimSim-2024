from typing import Any, Callable, Optional

import torch
from torch import nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


def to_sequential_features(x: torch.Tensor) -> torch.Tensor:
    """Convert the input tensor to sequential features.
    NOTE: This function is for ClimSim dataset only
    Checked with https://www.kaggle.com/code/abiolatti/keras-baseline-seq2seq
    Same results as keras implementation

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        x (torch.Tensor): The reshaped input tensor.
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
        scheduler_config: Optional[dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """CNN Constructor.

        Args:
            layers_hidden (Optional[nn.Sequential]): The hidden layers
            scheduler_config (Optional[dict[str, Any]]): The scheduler configuration
            loss_fn (Optional[Callable]): The loss function

        Attributes:
            _layers (nn.Sequential): The hidden layers
        """

        super().__init__(scheduler_config=scheduler_config, loss_fn=loss_fn)

        self._layers = layers_hidden or nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels=64, kernel_size=1, padding="same"),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 14, kernel_size=1, padding="same"),
        )

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        x = to_sequential_features(x)
        p_all = self._layers(x)

        p_seq = p_all[:, :6, :]
        p_seq = p_seq.permute(0, 2, 1).reshape(p_seq.shape[0], -1)

        p_flat = p_all[:, 6:14, :]
        p_flat = p_flat.mean(dim=2)

        P = torch.cat([p_seq, p_flat], dim=1)
        return P
