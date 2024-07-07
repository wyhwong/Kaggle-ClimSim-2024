import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import lightning
import torch
from torch import nn

import src.env
import src.logger
import src.pytorch.loss.r2
import src.schemas.constants


local_logger = src.logger.get_logger(__name__)


def cosine_decay_lr_scheduling(
    step: int,
    initial_lr: float,
    decay_steps: int,
    warmup_steps: int,
    alpha: float,
    maximum_lr: Optional[float] = None,
) -> float:
    """Learning rate schedule.
    This implementation is based on CosineDecay LR Schedule from TensorFlow.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay

    Args:
        step (int): The current step
        initial_lr (float): The initial learning rate
        decay_steps (int): The number of steps for decay
        warmup_steps (int): The number of steps for warmup
        alpha (float): The alpha parameter
        maximum_lr (float): The maximum learning rate

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


class ModelBase(lightning.LightningModule, ABC):
    """Multilayer perceptron model for regression."""

    def __init__(
        self,
        steps_per_epoch: int,
        loss_fn: Optional[Callable] = None,
    ) -> None:
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

        self._batch_loss: dict[str, list[float]] = {stage: [] for stage in src.schemas.constants.Stage}
        self._batch_r2: dict[str, list[float]] = {stage: [] for stage in src.schemas.constants.Stage}

        self._best_loss = {stage: float("inf") for stage in src.schemas.constants.Stage}
        self._best_r2 = {stage: -float("inf") for stage in src.schemas.constants.Stage}

    def __post_init__(self) -> None:
        """Post initialization."""

        warmup_steps = self._steps_per_epoch * src.env.N_WARMUP_EPOCHS
        decay_steps = self._steps_per_epoch * src.env.N_DECAY_EPOCHS

        self._optimizers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(self.parameters(), lr=1.0),
        ]
        self._schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_decay_lr_scheduling(
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

        local_logger.info("Initialized optimizer: %s.", self._optimizers[0].__class__.__name__)
        local_logger.info("Initialized scheduler: %s.", self._schedulers[0].__class__.__name__)
        local_logger.info("Initial Learning Rate: %.4f.", src.env.INITIAL_LR)
        local_logger.info("Maximum Learning Rate: %.4f.", src.env.MAXIMUM_LR)
        local_logger.info("Warmup Steps: %d.", warmup_steps)
        local_logger.info("Decay Steps: %d.", decay_steps)
        local_logger.info("Alpha: %.4f.", src.env.ALPHA)

    def replace_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    ) -> None:
        """Replace the optimizers and schedulers."""

        self._optimizers = optimizers
        self._schedulers = schedulers

    def _common_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        stage: src.schemas.constants.Stage,
    ) -> torch.Tensor:
        """Common step for training and validation."""

        # FIXME: Originally it should be X, y = batch
        # However, we are using a DataLoader with BatchSampler
        _x, _y = batch
        x, y = _x[0], _y[0]
        y_hat = self.forward(x)

        batch_loss = self._loss_fn(y_hat, y)
        batch_r2 = src.pytorch.loss.r2.r2_score_multivariate(y_hat, y)

        self.log(f"{stage}_loss", batch_loss)
        self.log(f"{stage}_r2", batch_r2)

        self._batch_loss[stage].append(batch_loss.item())
        self._batch_r2[stage].append(batch_r2.item())

        return batch_loss

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Training step."""

        batch_loss = self._common_step(batch, batch_idx, src.schemas.constants.Stage.TRAIN)
        return batch_loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> None:
        """Validation step."""

        self._common_step(batch, batch_idx, src.schemas.constants.Stage.VALID)

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> None:
        """Test step."""

        self._common_step(batch, batch_idx, src.schemas.constants.Stage.TEST)

    def _common_epoch_end(self, stage: src.schemas.constants.Stage) -> None:
        """Common epoch end for training and validation."""

        epoch_loss = sum(self._batch_loss[stage]) / len(self._batch_loss[stage])
        epoch_r2 = sum(self._batch_r2[stage]) / len(self._batch_r2[stage])

        self._batch_loss[stage].clear()
        self._batch_r2[stage].clear()

        self._best_loss[stage] = min(epoch_loss, self._best_loss[stage])
        self._best_r2[stage] = max(epoch_r2, self._best_r2[stage])

        local_logger.info(
            "Epoch %d - %s Best Loss: %.4f, Best R2: %.4f", self.current_epoch, stage, epoch_loss, epoch_r2
        )

        self.log(f"{stage}_epoch_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_epoch_r2", epoch_r2, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""

        self._common_epoch_end(src.schemas.constants.Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""

        self._common_epoch_end(src.schemas.constants.Stage.VALID)

    def configure_optimizers(self):
        """Configure the optimizers and schedulers."""

        return tuple(
            [
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",  # Step every batch
                        "frequency": 1,
                    },
                }
                for optimizer, scheduler in zip(self._optimizers, self._schedulers)
            ]
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
