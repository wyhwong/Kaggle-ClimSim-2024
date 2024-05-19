from typing import Callable, Optional

import torch
from torch import nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


class DynamicMLP(ModelBase):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        layers: Optional[nn.Sequential] = None,
        loss_train: Optional[Callable] = None,
        loss_val: Optional[Callable] = None,
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
            layers (Optional[nn.Sequential]): Custom layers for the model
            loss_train (Optional[nn.modules.loss._Loss]): Loss function for training
            loss_val (Optional[nn.modules.loss._Loss]): Loss function for validation

        Returns:
            None
        """

        super().__init__(
            loss_train=loss_train,
            loss_val=loss_val,
        )

        self._layers = layers or nn.Sequential(
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        return self._layers(X)
