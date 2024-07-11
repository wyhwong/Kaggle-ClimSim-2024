import datetime
from typing import Optional

import lightning
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import src.logger


local_logger = src.logger.get_logger(__name__)


def get_default_trainer(
    deterministic: bool,
    model_name: str,
    max_epochs: int = 100,
    max_time: datetime.timedelta = datetime.timedelta(days=3),
    log_every_n_steps=10,
    check_val_every_n_epoch: int = 3,
    callbacks: Optional[list[lightning.pytorch.callbacks.Callback]] = None,
    use_distributed_sampler: bool = False,
    random_seed: int = 2024,
    **kwargs,
) -> lightning.pytorch.Trainer:
    """Get the default trainer for the model.

    Args:
        deterministic (bool): Whether to use deterministic training
        model_name (str): The name of the model
        max_epochs (int, optional): The maximum number of epochs. Defaults to 100.
        max_time (datetime.timedelta, optional): The maximum time for training. Defaults to datetime.timedelta(hours=1).
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        callbacks (Optional[list[lightning.pytorch.callbacks.Callback]], optional): The callbacks. Defaults to None.
        use_distributed_sampler (bool, optional): Use distributed sampler. Defaults to False.
        random_seed (int, optional): The random seed. Defaults to 2024.

    Returns:
        lightning.pytorch.Trainer: The trainer object
    """

    lightning.pytorch.seed_everything(random_seed, workers=True)

    base_dirpath = "./models"
    version = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
                dirpath=f"{base_dirpath}/{model_name}/{version}",
                filename=model_name + "-best-{epoch:02d}",
                save_top_k=1,
                save_last=True,
                verbose=True,
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=f"{base_dirpath}/{model_name}/{version}",
                filename=model_name + "-{epoch:02d}",
                every_n_epochs=5,
                save_top_k=-1,
                verbose=True,
            ),
        ]

    trainer = lightning.pytorch.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,
        deterministic=deterministic,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=f"./logs/{model_name}/{version}",
        callbacks=callbacks,
        use_distributed_sampler=use_distributed_sampler,
        **kwargs,
    )
    return trainer