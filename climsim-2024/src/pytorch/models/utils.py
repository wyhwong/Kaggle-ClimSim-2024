import datetime
from typing import Optional

import lightning
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


def get_default_trainer(
    model_name: str,
    max_epochs: int = 100,
    max_time: datetime.timedelta = datetime.timedelta(hours=1),
    log_every_n_steps=10,
    check_val_every_n_epoch: int = 3,
    callbacks: Optional[list[lightning.pytorch.callbacks.Callback]] = None,
    use_distributed_sampler: bool = False,
    **kwargs,
) -> lightning.pytorch.Trainer:
    """
    Get the default trainer for the model.

    Args:
        model_name (str): The name of the model
        max_epochs (int, optional): The maximum number of epochs. Defaults to 100.
        max_time (datetime.timedelta, optional): The maximum time for training. Defaults to datetime.timedelta(hours=1).
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        callbacks (Optional[list[lightning.pytorch.callbacks.Callback]], optional): The callbacks. Defaults to None.
        use_distributed_sampler (bool, optional): Use distributed sampler. Defaults to False.

    Returns:
        lightning.pytorch.Trainer: The trainer object
    """

    lightning.pytorch.seed_everything(src.env.RANDOM_SEED, workers=True)
    if callbacks is None:
        callbacks = [
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=True,
                log_weight_decay=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=True,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="./models",
                filename=model_name,
                save_top_k=1,
                verbose=True,
                mode="min",
            ),
        ]

    trainer = lightning.pytorch.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,
        deterministic=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=f"./training_logs/{model_name}",
        callbacks=callbacks,
        use_distributed_sampler=use_distributed_sampler,
        **kwargs,
    )
    return trainer
