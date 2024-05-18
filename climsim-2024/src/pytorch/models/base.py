from abc import ABC, abstractmethod
from typing import Callable, Optional

import lightning
import torch
from torch import nn

import src.logger


local_logger = src.logger.get_logger(__name__)


class ModelBase(lightning.LightningModule, ABC):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        loss_train: Optional[Callable] = None,
        loss_val: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the model.

        Args:
            loss_train: Loss function for training
            loss_val: Loss function for validation

        Returns:
            None
        """

        super().__init__()

        self._loss_train = loss_train or nn.functional.mse_loss
        self._loss_val = loss_val or nn.functional.mse_loss

    def __post_init__(self) -> None:
        """Post initialization."""

        self._optimizers: list[torch.optim.Optimizer] = [torch.optim.Adam(self.parameters(), lr=1e-4)]
        self._schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) for optimizer in self._optimizers
        ]

    def replace_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    ) -> None:
        """Set the optimizers."""

        self._optimizers = optimizers
        self._schedulers = schedulers

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Training step."""

        _X, _y = batch
        X, y = _X[0], _y[0]
        y_hat = self.forward(X)
        loss = self._loss_train(y_hat, y)

        local_logger.debug("Training loss: %4f", loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Validation step."""

        _X, _y = batch
        X, y = _X[0], _y[0]
        y_hat = self.forward(X)
        loss = self._loss_val(y_hat, y)

        local_logger.debug("Validation loss: %4f", loss)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Return the optimizer."""

        return self._optimizers, self._schedulers
