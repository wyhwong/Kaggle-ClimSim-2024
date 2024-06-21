from abc import ABC, abstractmethod
from typing import Callable, Optional

import lightning
import torch
from torch import nn

import src.logger


local_logger = src.logger.get_logger(__name__)


class ModelBase(lightning.LightningModule, ABC):
    """Multilayer perceptron model for regression."""

    def __init__(self, loss_fn: Optional[Callable] = None) -> None:
        """
        Initialize the model.

        Args:
            loss_train: Loss function for training
            loss_val: Loss function for validation

        Returns:
            None
        """

        super().__init__()

        self._loss_fn = loss_fn or nn.functional.mse_loss

        self._batch_loss_train: list[float] = []
        self._batch_loss_val: list[float] = []

        self._epoch_loss_train: dict[int, float] = {}
        self._epoch_loss_val: dict[int, float] = {}

        self._best_loss_train: float = float("inf")
        self._best_loss_val: float = float("inf")

    def __post_init__(self) -> None:
        """Post initialization."""

        self._optimizers: list[torch.optim.Optimizer] = [torch.optim.AdamW(self.parameters(), lr=1e-4)]
        self._schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
            # torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99) for optimizer in self._optimizers
            # torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=600)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=600, eta_min=1e-7)
            for optimizer in self._optimizers
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # FIXME: Originally it should be x, y = batch
        # However, we are using a DataLoader with BatchSampler
        _x, _y = batch
        x, y = _x[0], _y[0]
        y_hat = self.forward(x)
        batch_loss = self._loss_fn(y_hat, y)

        self._batch_loss_train.append(batch_loss.detach().cpu().numpy())
        self.log("train_loss", batch_loss)
        return batch_loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""

        epoch_loss = sum(self._batch_loss_train) / len(self._batch_loss_train)
        self._epoch_loss_train[self.current_epoch] = epoch_loss
        self._batch_loss_train.clear()

        if epoch_loss < self._best_loss_train:
            self._best_loss_train = epoch_loss
            local_logger.info("Epoch %d - Best Training Loss: %.4f", self.current_epoch, epoch_loss)
        else:
            local_logger.info("Epoch %d - Training Loss: %.4f", self.current_epoch, epoch_loss)

        self.log("train_epoch_loss", epoch_loss, on_step=False, on_epoch=True)

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Validation step."""

        # FIXME: Originally it should be X, y = batch
        # However, we are using a DataLoader with BatchSampler
        _x, _y = batch
        x, y = _x[0], _y[0]
        y_hat = self.forward(x)
        batch_loss = self._loss_fn(y_hat, y)

        self._batch_loss_val.append(batch_loss.detach().cpu().numpy())
        self.log("val_loss", batch_loss)
        return batch_loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""

        epoch_loss = sum(self._batch_loss_val) / len(self._batch_loss_val)
        self._epoch_loss_val[self.current_epoch] = epoch_loss
        self._batch_loss_val.clear()

        if epoch_loss < self._best_loss_val:
            self._best_loss_val = epoch_loss
            local_logger.info("Epoch %d - Best Validation Loss: %.4f", self.current_epoch, epoch_loss)
        else:
            local_logger.info("Epoch %d - Validation Loss: %.4f", self.current_epoch, epoch_loss)

        self.log("val_epoch_loss", epoch_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Return the optimizer."""

        return self._optimizers, self._schedulers
