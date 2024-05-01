import torch
from torch import nn

import logger


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
