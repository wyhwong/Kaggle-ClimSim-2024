from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.schemas.visualization as sv
import src.visualization.base as base


def loss_curve(
    losses: list[pd.Series],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close=True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the training/validation loss curve against the number of epochs.

    Args:
        losses (list[pd.Series]): List of training/validation loss
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
    for ds_loss in losses:
        sns.lineplot(data=ds_loss, ax=ax)
    ax.set(ylabel="", xlabel="")
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)
