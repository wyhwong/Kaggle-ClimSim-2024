import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import lightning
import torch
from torch import nn

import src.env
import src.logger
import src.pytorch.loss.r2


local_logger = src.logger.get_logger(__name__)


class ModelBase(lightning.LightningModule, ABC):
    """Multilayer perceptron model for regression."""

    def __init__(self, steps_per_epoch: int, loss_fn: Optional[Callable] = None) -> None:
        """
        Initialize the model.

        Args:
            steps_per_epoch (int): Number of steps per epoch
            loss_fn (Optional[Callable], optional): Loss function. Defaults to None.

        Returns:
            None
        """

        super().__init__()

        self._steps_per_epoch = steps_per_epoch
        self._loss_fn = loss_fn or nn.functional.mse_loss

        self._batch_loss_train: list[float] = []
        self._batch_loss_val: list[float] = []
        self._batch_r2_loss: list[float] = []

        self._epoch_loss_train: dict[int, float] = {}
        self._epoch_loss_val: dict[int, float] = {}

        self._best_loss_train: float = float("inf")
        self._best_loss_val: float = float("inf")

    def __post_init__(self) -> None:
        """Post initialization."""

        def lr_lambda(
            step: int,
            initial_lr: float,
            maximum_lr: float,
            decay_steps: int,
            warmup_steps: int,
            alpha: float,
        ) -> float:
            """Learning rate schedule.
            This implementation is based on CosineDecay LR Schedule from TensorFlow.
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay

            Args:
                step (int): The current step
                initial_lr (float): The initial learning rate
                maximum_lr (float): The maximum learning rate
                decay_steps (int): The number of steps for decay
                warmup_steps (int): The number of steps for warmup
                alpha (float): The alpha parameter

            Returns:
                float: The learning rate
            """

            if maximum_lr is None:
                maximum_lr = initial_lr

            if step < warmup_steps:
                # Linear warmup phase
                completed_fraction = step / warmup_steps
                total_delta = maximum_lr - initial_lr
                return initial_lr + completed_fraction * total_delta
            else:
                # Cosine decay phase
                step_in_decay_phase = step - warmup_steps
                step_in_decay_phase = min(step_in_decay_phase, decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay_phase / decay_steps))
                decayed = (1 - alpha) * cosine_decay + alpha
                return maximum_lr * decayed

        warmup_steps = self._steps_per_epoch * src.env.N_WARMUP_EPOCHS
        decay_steps = self._steps_per_epoch * src.env.N_DECAY_EPOCHS

        local_logger.info("Initial Learning Rate: %.4f", src.env.INITIAL_LR)
        local_logger.info("Maximum Learning Rate: %.4f", src.env.MAXIMUM_LR)
        local_logger.info("Warmup Steps: %d", warmup_steps)
        local_logger.info("Decay Steps: %d", decay_steps)
        local_logger.info("Alpha: %.4f", src.env.ALPHA)

        self._optimizers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(self.parameters(), lr=1.0),
        ]
        self._schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: lr_lambda(
                    step=step,
                    initial_lr=src.env.INITIAL_LR,
                    maximum_lr=src.env.MAXIMUM_LR,
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    alpha=src.env.ALPHA,
                ),
            )
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
        """Forward pass of the model."""

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

        r2_loss = src.pytorch.loss.r2.r2_score_multivariate(y_hat, y)
        self._batch_r2_loss.append(r2_loss.detach().cpu().numpy())
        self.log("r2_loss", r2_loss)

        return batch_loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""

        epoch_loss = sum(self._batch_loss_val) / len(self._batch_loss_val)
        self._epoch_loss_val[self.current_epoch] = epoch_loss
        self._batch_loss_val.clear()

        epoch_r2_loss = sum(self._batch_r2_loss) / len(self._batch_r2_loss)
        self._batch_r2_loss.clear()

        if epoch_loss < self._best_loss_val:
            self._best_loss_val = epoch_loss
            local_logger.info("Epoch %d - Best Validation Loss: %.4f", self.current_epoch, epoch_loss)
        else:
            local_logger.info("Epoch %d - Validation Loss: %.4f", self.current_epoch, epoch_loss)

        self.log("val_epoch_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log("epoch_r2_loss", epoch_r2_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Return the optimizer."""

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "step",  # Step every batch
                "frequency": 1,
            }
            for scheduler in self._schedulers
        ]

        return self._optimizers, schedulers
