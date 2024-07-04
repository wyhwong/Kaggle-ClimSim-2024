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
        layers_hidden: Optional[nn.Sequential] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a multilayer perceptron model for regression.
        NOTE: If layers is not provided, the default layers are used.
              For default, we have the baseline MLP model in the paper.
              The layer content are:
                - Input layer: 556 neurons (Input features of ClimSim Dataset)
                - Activation function: LeakyReLU
                - Hidden layer 1: 768 neurons
                - Activation function: LeakyReLU
                - Hidden layer 2: 640 neurons
                - Activation function: LeakyReLU
                - Hidden layer 3: 512 neurons
                - Activation function: LeakyReLU
                - Hidden layer 4: 640 neurons
                - Activation function: LeakyReLU
                - Hidden layer 5: 640 neurons
                - Activation function: LeakyReLU
                - Output layer: 368 neuron (Output target of ClimSim Dataset)

        Args:
            steps_per_epoch (int): Number of steps per epoch
            layers (Optional[nn.Sequential]): Custom layers for the model
            loss_fn (Optional[Callable]): Loss function for training

        Returns:
            None
        """

        super().__init__(steps_per_epoch=steps_per_epoch, loss_fn=loss_fn)

        self._layers = layers_hidden or nn.Sequential(
            nn.Linear(556, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 640),
            nn.LeakyReLU(),
            nn.Linear(640, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 640),
            nn.LeakyReLU(),
            nn.Linear(640, 640),
            nn.LeakyReLU(),
            nn.Linear(640, 368),
        )

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        return self._layers(x)
