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
        layers_hidden: Optional[list[int]] = None,
        kernal_size_hidden: Optional[list[int]] = None,
        padding: str = "same",
        scheduler_config: Optional[dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """CNN Constructor.

        Args:
            layers_hidden (list[int]): Hidden layer sizes
            kernal_size_hidden (list[int]): Kernel sizes for hidden layers
            padding (str): Padding type
            scheduler_config (dict[str, Any]): Scheduler configuration
            loss_fn (Callable): Loss function

        raises:
            ValueError: If the number of hidden layers is not one less than the number of kernel sizes

        Attributes:
            _layers (nn.Sequential): The hidden layers
        """

        super().__init__(scheduler_config=scheduler_config, loss_fn=loss_fn)

        if layers_hidden is None:
            layers_hidden = [25, 128, 64, 14]

        if kernal_size_hidden is None:
            kernal_size_hidden = [3, 3, 3]

        if len(layers_hidden) - 1 != len(kernal_size_hidden):
            message = "The number of hidden layers should be one less than the number of kernel sizes."
            local_logger.error(message)
            raise ValueError(message)

        layers: list[nn.Module] = []
        for i in range(len(layers_hidden) - 2):
            layers.append(
                nn.Conv1d(
                    in_channels=layers_hidden[i],
                    out_channels=layers_hidden[i + 1],
                    kernel_size=kernal_size_hidden[i],
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layers_hidden[i + 1]))

        layers.append(
            nn.Conv1d(
                in_channels=layers_hidden[-2],
                out_channels=layers_hidden[-1],
                kernel_size=kernal_size_hidden[-1],
                padding=padding,
            )
        )
        self._layers = nn.Sequential(*layers)

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
