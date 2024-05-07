from typing import Optional

import lightning
import torch
from torch import nn

import src.logger


local_logger = src.logger.get_logger(__name__)


# TODO: Remove deprecated version of MLP model
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


class DynamicMLP(lightning.LightningModule):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        layers: Optional[nn.Sequential] = None,
        loss_train: Optional[nn.modules.loss._Loss] = None,
        loss_val: Optional[nn.modules.loss._Loss] = None,
        optimizers: Optional[list[torch.optim.Optimizer]] = None,
        schedulers: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None,
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
            optimizers (Optional[list[torch.optim.Optimizer]]): Optimizers for the model
            schedulers (Optional[list[torch.optim.lr_scheduler.LRScheduler]]): Learning rate schedulers

        Returns:
            None
        """

        super().__init__()

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
        self._loss_train = loss_train or nn.functional.mse_loss
        self._loss_val = loss_val or nn.functional.mse_loss
        self._optimizers = optimizers or [torch.optim.Adam(self.parameters(), lr=1e-4)]
        self._scheduler = schedulers or [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) for optimizer in self._optimizers
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        return self._layers(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Training step."""

        x, y = batch
        y_hat = self.forward(x)
        loss = self._loss_train(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Validation step."""

        x, y = batch
        y_hat = self.forward(x)
        loss = self._loss_val(y_hat, y)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Return the optimizer."""

        return self._optimizers, self._scheduler
