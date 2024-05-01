import datetime
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import env
import logger
import schemas.constants as sc


local_logger = logger.get_logger(__name__)


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


def train(
    model: nn.Module,
    dataloaders: dict[sc.Phase, torch.utils.data.DataLoader],
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    loss_fns: Optional[dict[sc.Phase, torch.nn.modules.loss._Loss]] = None,
) -> tuple[nn.Module, dict[str, torch.Tensor], dict[sc.Phase, list[float]]]:
    """
    Train model

    Args:
        model (nn.Module): Model structure
        dataloaders (dict[Phase, DataLoader]): Data loaders
        num_epochs (int): Number of epochs
        optimizier (Optimizer): Optimizer
        scheduler (Optional[LRScheduler]): Learning rate scheduler
        loss_fns (Optional[dict[Phase, _Loss]]): Loss functions

    Returns:
        tuple: Model, best weights, loss
    """

    # Initialize loss functions if not provided
    if loss_fns is None:
        loss_fns = {sc.Phase.TRAINING: torch.nn.L1Loss(), sc.Phase.VALIDATION: torch.nn.L1Loss()}

    # Initialize the best weights and the best validation loss
    best_weights = deepcopy(model.state_dict())
    loss: dict[sc.Phase, list[float]] = {sc.Phase.TRAINING: [], sc.Phase.VALIDATION: []}
    best_val_loss = np.inf

    training_start = datetime.datetime.now()
    local_logger.info("Start time of training: %s", training_start)
    local_logger.info("Training using device: %s", env.DEVICE)

    model.to(env.DEVICE)

    for epoch in range(1, num_epochs + 1):
        # Logging
        local_logger.info("-" * 40)
        local_logger.info("Epoch %d/%d", epoch, num_epochs)
        local_logger.info("-" * 20)

        for phase in [sc.Phase.TRAINING, sc.Phase.VALIDATION]:
            local_logger.debug("The %d-th epoch %s started.", epoch, phase)
            if phase is sc.Phase.TRAINING:
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            # Iterate over data
            for inputs, outputs in tqdm(dataloaders[phase]):
                inputs = inputs.to(env.DEVICE)
                outputs = outputs.to(env.DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase is sc.Phase.TRAINING):
                    predictions = model(inputs.float())
                    _loss = loss_fns[phase](predictions, outputs)

                    if phase is sc.Phase.TRAINING:
                        _loss.backward()
                        optimizer.step()

                epoch_loss += _loss.item()

            if scheduler and phase is sc.Phase.TRAINING:
                scheduler.step()
                local_logger.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])

            local_logger.info("[%s] Loss: %.4f.", phase, epoch_loss)

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
