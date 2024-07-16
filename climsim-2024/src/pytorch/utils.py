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
    NOTE: By default, the trainer will:
        - seed_everything
        - log learning rate, momentum, and weight decay
        - early stop based on validation loss
        - save the best model based on validation loss
        - save the model every 5 epochs
        - log to ./logs/{model_name}/{version}
        - save the model to ./models/{model_name}/{version}

    Args:
        deterministic (bool): Whether to set the random seed
        model_name (str): The model name
        max_epochs (int): The maximum number of epochs
        max_time (datetime.timedelta): The maximum time
        log_every_n_steps (int): The number of steps to log
        check_val_every_n_epoch (int): The number of epochs to check validation
        callbacks (Optional[list[lightning.pytorch.callbacks.Callback]]): The callbacks
        use_distributed_sampler (bool): Whether to use distributed sampler
        random_seed (int): The random seed
        **kwargs: Additional

    Returns:
        trainer (lightning.pytorch.Trainer): The trainer
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
