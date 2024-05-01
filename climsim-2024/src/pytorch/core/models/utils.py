import torch
from torch import nn

import pytorch.logger


local_logger = pytorch.logger.get_logger(__name__)


def load_model(model: nn.Module, model_path: str) -> None:
    """
    Load model from file.

    Args:
        model (nn.Module): Model structure
        model_path (str): Model file path

    Returns:
        None
    """

    local_logger.trace(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    local_logger.info(f"Model loaded from {model_path}")


def save_weights(model: nn.Module, model_path: str) -> None:
    """
    Save model weights to file.

    Args:
        model (nn.Module): Model structure
        model_path (str): Model file path

    Returns:
        None
    """

    local_logger.trace(f"Saving model weights to {model_path}...")
    torch.save(model.state_dict(), model_path)
    local_logger.info(f"Model weights saved to {model_path}")
