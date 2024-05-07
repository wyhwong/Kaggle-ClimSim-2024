import datetime
from copy import deepcopy
from typing import Optional

import lightning
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import src.env
import src.logger
import src.schemas.constants as sc


local_logger = src.logger.get_logger(__name__)


# TODO: Remove this deprecated function
def load_model(model: nn.Module, model_path: str) -> None:
    """
    Load model from file.

    Args:
        model (nn.Module): Model structure
        model_path (str): Model file path

    Returns:
        None
    """

    local_logger.debug("Loading model from %s...", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    local_logger.info("Model loaded from %s.", model_path)


# TODO: Remove this deprecated function
def save_weights(model: nn.Module, model_path: str) -> None:
    """
    Save model weights to file.

    Args:
        model (nn.Module): Model structure
        model_path (str): Model file path

    Returns:
        None
    """

    local_logger.debug("Saving model weights to %s...", model_path)
    torch.save(model.state_dict(), model_path)
    local_logger.info("Model weights saved to %s.", model_path)


# TODO: Remove this deprecated function
def train(
    model: nn.Module,
    dataloaders: dict[sc.Phase, torch.utils.data.DataLoader],
    n_epochs: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    loss_fns: Optional[dict[sc.Phase, torch.nn.modules.loss._Loss]] = None,
    validate_every_n_epoch: int = 3,
) -> tuple[nn.Module, dict[str, torch.Tensor], dict[sc.Phase, list[float]]]:
    """
    Train model

    Args:
        model (nn.Module): Model structure
        dataloaders (dict[Phase, DataLoader]): Data loaders
        num_epochs (int): Number of epochs
        optimizier (Optional[Optimizer]): Optimizer
        scheduler (Optional[LRScheduler]): Learning rate scheduler
        loss_fns (Optional[dict[Phase, _Loss]]): Loss functions
        validate_every_n_epoch (int): Validate every n epoch

    Returns:
        tuple: Model, best weights, loss
    """

    # Seed reset (for reproducibility)
    torch.manual_seed(src.env.RANDOM_SEED)
    np.random.seed(src.env.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    local_logger.info("Random seed set to %d.", src.env.RANDOM_SEED)

    # Initialize loss functions if not provided
    if loss_fns is None:
        local_logger.info("Loss functions not provided. Using L1Loss.")
        loss_fns = {sc.Phase.TRAINING: torch.nn.L1Loss(), sc.Phase.VALIDATION: torch.nn.L1Loss()}

    # Initialize Adam optimizer if not provided
    if optimizer is None:
        local_logger.info("Optimizer not provided. Using Adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the best weights and the best validation loss
    best_weights = deepcopy(model.state_dict())
    loss: dict[sc.Phase, list[float]] = {sc.Phase.TRAINING: [], sc.Phase.VALIDATION: []}
    best_val_loss = np.inf

    training_start = datetime.datetime.now()
    local_logger.info("Start time of training: %s", training_start)
    local_logger.info("Training using device: %s", src.env.DEVICE)

    model.to(src.env.DEVICE)

    for epoch in range(1, n_epochs + 1):
        # Logging
        local_logger.info("-" * 40)
        local_logger.info("Epoch %d/%d", epoch, n_epochs)
        local_logger.info("-" * 20)

        for phase in [sc.Phase.TRAINING, sc.Phase.VALIDATION]:
            epoch_loss = _compute_epoch_loss_and_update_weights(
                epoch=epoch,
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                loss_fns=loss_fns,
                validate_every_n_epoch=validate_every_n_epoch,
                scheduler=scheduler,
            )

            if phase is sc.Phase.VALIDATION and epoch_loss < best_val_loss:
                local_logger.info("New Record: %.4f < %.4f", epoch_loss, best_val_loss)
                best_val_loss = epoch_loss
                best_weights = deepcopy(model.state_dict())
                local_logger.debug("Updated best models.")

            loss[phase].append(float(epoch_loss))

    time_elapsed = (datetime.datetime.now() - training_start).total_seconds()
    local_logger.info("Training complete at %s", datetime.datetime.now())
    local_logger.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
    local_logger.info("Best val: %.4f}.", best_val_loss)
    return (model, best_weights, loss)


# TODO: Remove this deprecated function
def _compute_epoch_loss_and_update_weights(
    epoch: int,
    model: nn.Module,
    dataloaders: dict[sc.Phase, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fns: dict[sc.Phase, torch.nn.modules.loss._Loss],
    validate_every_n_epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    """
    Compute loss and update weights.

    Args:
        epoch (int): Epoch number
        model (nn.Module): Model structure
        dataloaders (dict[Phase, DataLoader]): Data loaders
        optimizer (Optimizer): Optimizer
        loss_fns (dict[Phase, _Loss]): Loss functions
        validate_every_n_epoch (int): Validate every n epoch
        scheduler (Optional[LRScheduler]): Learning rate scheduler

    Returns:
        float: Loss
    """

    for phase in [sc.Phase.TRAINING, sc.Phase.VALIDATION]:
        local_logger.debug("The %d-th epoch %s started.", epoch, phase)
        if phase is sc.Phase.TRAINING:
            model.train()
        elif phase is sc.Phase.VALIDATION and epoch % validate_every_n_epoch == 0:
            model.eval()
        else:
            local_logger.debug("Skipping validation in epoch %d.", epoch)
            continue

        epoch_loss = 0.0
        # Iterate over data
        for inputs, outputs in tqdm(dataloaders[phase]):
            # NOTE: Here we comment out the conversion to device
            #       because the Dataset in parquet.py already
            #       handles this conversion
            # inputs = inputs.to(src.env.DEVICE)
            # outputs = outputs.to(src.env.DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase is sc.Phase.TRAINING):
                predictions = model(inputs)
                _loss = loss_fns[phase](predictions, outputs)

                if phase is sc.Phase.TRAINING:
                    _loss.backward()
                    optimizer.step()

            epoch_loss += _loss.item()

        if scheduler and phase is sc.Phase.TRAINING:
            scheduler.step()
            local_logger.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])

    local_logger.info("[%s] Loss: %.4f.", phase, epoch_loss)
    return epoch_loss


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
