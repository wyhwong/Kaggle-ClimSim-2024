import numpy as np


class Dataset:
    """
    Dataset class for PyTorch dataloader

    TODO: Replace this class with a better solution:
        1. allow partial loading of dataset (probably polars)
        2. sampling method (TBC, optional)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset with input and output data

        Args:
            X (np.array): Input data
            y (np.array): Output data

        Returns:
            None
        """

        self.X = X
        self.y = y

    def __len__(self):
        """Return the length of the dataset"""

        return len(self.X)

    def __getitem__(self, idx):
        """Return the data at the given index"""

        return self.X[idx], self.y[idx]
