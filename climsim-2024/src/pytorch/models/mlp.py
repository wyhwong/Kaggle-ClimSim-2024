from typing import Any, Callable, Optional

import torch
from torch import nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


class MLP(ModelBase):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        layers_hidden: Optional[list[int]] = None,
        scheduler_config: Optional[dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """MLP Constructor.

        Args:
            layers_hidden: hidden layer sizes
            scheduler_config: scheduler configuration
            loss_fn: loss function

        Attributes:
            _layers: hidden layers

        Methods:
            forward: Forward pass through the network
        """

        if layers_hidden is None:
            layers_hidden = [556, 1024, 512, 368]

        super().__init__(scheduler_config=scheduler_config, loss_fn=loss_fn)

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
