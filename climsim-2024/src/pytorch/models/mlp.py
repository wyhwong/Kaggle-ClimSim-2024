from typing import Callable, Optional

import torch
from torch import nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


class MLP(ModelBase):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        steps_per_epoch: int,
        layers_hidden: list[int],
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a multilayer perceptron model for regression.

        Args:
            steps_per_epoch (int): Number of steps per epoch
            layers_hidden (list[int]): Number of hidden units in each layer
            loss_fn (Optional[Callable]): Loss function for training

        Returns:
            None
        """

        super().__init__(steps_per_epoch=steps_per_epoch, loss_fn=loss_fn)

        layers: list[nn.Module] = []
        for i in range(len(layers_hidden) - 2):
            layers.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            layers.append(nn.LayerNorm(layers_hidden[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layers_hidden[-2], layers_hidden[-1]))
        self._layers = nn.Sequential(*layers)

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        return self._layers(x)
