import math

import torch


class CosineDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """Cosine Decay Learning Rate Scheduler.
    This implementation is based on CosineDecay LR Schedule from TensorFlow.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr=1e-4,
        maximum_lr=1e-3,
        alpha=0.1,
        warmup_epochs=1,
        decay_epochs=100,
        steps_per_epoch=1000,
        last_epoch=-1,
        verbose=False,
    ) -> None:
        """Cosine Decay Learning Rate Scheduler."""

        self._initial_lr = initial_lr
        self._maximum_lr = maximum_lr
        self._alpha = alpha
        self._warmup_steps = warmup_epochs * steps_per_epoch
        self._decay_steps = decay_epochs * steps_per_epoch

        super().__init__(
            optimizer=optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def _lr_lambda(self, step: int) -> float:
        """Learning rate function.

        Args:
            step (int): The current step

        Returns:
            float: The learning rate
        """

        if self._maximum_lr is None:
            maximum_lr = self._initial_lr

        if step < self._warmup_steps:
            # Linear warmup phase
            completed_fraction = step / self._warmup_steps
            total_delta = maximum_lr - self._initial_lr
            return self._initial_lr + completed_fraction * total_delta
        else:
            # Cosine decay phase
            step_in_decay_phase = step - self._warmup_steps
            step_in_decay_phase = min(step_in_decay_phase, self._decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay_phase / self._decay_steps))
            decayed = (1 - self._alpha) * cosine_decay + self._alpha
            return maximum_lr * decayed
