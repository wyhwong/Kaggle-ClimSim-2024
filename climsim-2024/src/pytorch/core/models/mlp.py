from typing import Optional

import torch
from torch import nn

import src.logger


local_logger = src.logger.get_logger(__name__)


class MLP(nn.Module):
    """Multilayer perceptron model for regression."""

    def __init__(self, layers: Optional[nn.Sequential] = None) -> None:
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

        Returns:
            None
        """

        super().__init__()

        if layers is not None:
            self.layers = layers
        else:
            self.layers = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (nn.Tensor): Input tensor

        Returns:
            nn.Tensor: Output tensor
        """

        return self.layers(x)
