import datetime

import lightning

import src.env
import src.logger


local_logger = src.logger.get_logger(__name__)


def get_default_trainer(
    max_epochs: int = 100,
    max_time: datetime.timedelta = datetime.timedelta(hours=1),
    check_val_every_n_epoch: int = 3,
    **kwargs,
) -> lightning.pytorch.Trainer:
    """
    Get the default trainer for the model.

    Args:
        max_epochs (int): Maximum number of epochs to train
        max_time (datetime.timedelta): Maximum time to train
        check_val_every_n_epoch (int): Check validation every n epochs
        **kwargs: Additional arguments to pass to the trainer

    Returns:
        lightning.pytorch.Trainer: The trainer object
    """

    lightning.pytorch.seed_everything(src.env.RANDOM_SEED, workers=True)
    trainer = lightning.pytorch.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,
        deterministic=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        **kwargs,
    )
    return trainer
