from typing import Optional

import seaborn as sns
import matplotlib.pyplot as plt

import visualization.base as base
import schemas.visualization as sv


def loss_curve(
    loss: dict[str, list[float]],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close=True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the training/validation loss curve against the number of epochs.

    Args:
        loss (dict[str, list[float]]): Training/Validation loss
        filename (str): Filename
        output_dir (str): Output directory
        close (bool): Close figure

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes
    """

    labels = sv.Labels(
        title="Training/Validation Loss against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Loss",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=loss, ax=ax)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)
