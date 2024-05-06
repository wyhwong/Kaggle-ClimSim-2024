from typing import Optional

import lightning
import torch
from torch import nn

import src.logger


local_logger = src.logger.get_logger(__name__)


class DynamicNetworkInferface(lightning.LightningModule):
    """Dynamic network interface for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        loss_train: Optional[nn.modules.loss._Loss] = None,
        loss_val: Optional[nn.modules.loss._Loss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Initialize the model.

        Args:
            model: The model to be trained.
            loss_train: The loss function to be used during training.
            loss_val: The loss function to be used during validation.
            optimizer: The optimizer to be used during training.

        Returns:
            None
        """

        super().__init__()
        self._model = model

        self._optimizer = optimizer
        self._loss_train = loss_train or nn.L1Loss()
        self._loss_val = loss_val or nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        return self.model.forward(x)

    def training_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        """Training step."""

        x, y = batch
        y_hat = self.model(x)
        loss = self._loss_train(y_hat, y)

        return loss

    def validation_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        """Validation step."""

        x, y = batch
        y_hat = self.model(x)
        loss = self._loss_val(y_hat, y)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer."""

        if self.optimizer is None:
            local_logger.info("Optimizer not provided. Using Adam optimizer.")
            optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
            return optimizer

        return self.optimizer

    def to_trainer(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        **kwargs,
    ) -> lightning.Trainer:
        """Return a trainer for the model."""

        return lightning.Trainer(self, train_loader, val_loader, **kwargs)
