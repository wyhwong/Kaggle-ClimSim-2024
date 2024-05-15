import lightning
import torch

import src.logger


local_logger = src.logger.get_logger(__name__)


class Loadermodule(lightning.LightningDataModule):
    """Dynamic dataloader interface for PyTorch models."""

    def __init__(
        self,
        loader_train: torch.utils.data.DataLoader,
        loader_val: torch.utils.data.DataLoader,
    ) -> None:
        """
        Initialize the dataloader interface.

        Args:
            loader_train (torch.utils.data.DataLoader): Training dataloader
            loader_val (torch.utils.data.DataLoader): Validation dataloader

        Returns:
            None
        """

        super().__init__()
        self._loader_train = loader_train
        self._loader_val = loader_val

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""

        return self._loader_train

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader."""

        return self._loader_val
