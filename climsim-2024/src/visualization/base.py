from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

import logger
import schemas.visualization as sv
import utils


local_logger = logger.get_logger(__name__)


def initialize_plot(
    nrows=1,
    ncols=1,
    figsize=(10, 6),
    labels=sv.Labels(),
    padding=sv.Padding(),
    fontsize=12,
    lines=Optional[list[sv.Line]],
) -> tuple[plt.Figure, plt.Axes]:
    """
    Initialize plot with specified number of rows and columns.

    Args:
        nrows (int): Number of rows
        ncols (int): Number of columns
        figsize (tuple): Figure size
        labels (Labels): Labels
        padding (Padding): Padding
        fontsize (int): Font size
        lines (list[Line]): Lines to be added to the plot

    Returns:
        tuple: Figure and axes
    """

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )
    local_logger.debug("Initialized plot: nrows=%d, ncols=%d, figsize=%s.", nrows, ncols, figsize)

    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    local_logger.debug("Adjusted plot: tpad=%f, lpad=%f, bpad=%f.", padding.tpad, padding.lpad, padding.bpad)

    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(x=0.04, y=0.5, s=labels.ylabel, fontsize=fontsize, rotation="vertical", verticalalignment="center")
    fig.text(x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center")
    local_logger.debug(
        "Added title and labels: title=%s, xlabel=%s, ylabel=%s.", labels.title, labels.xlabel, labels.ylabel
    )

    if lines is not None:
        add_lines_to_plot(axes, lines)
        local_logger.debug("Added lines to plot.")

    return (fig, axes)


def add_lines_to_plot(ax: plt.Axes, lines: list[sv.Line]) -> None:
    """
    Add lines to plot.

    Args:
        ax (plt.Axes): Axes
        lines (list[Line]): Lines

    Returns:
        None

    Raises:
        TypeError: If line is not an instance of Line
    """

    for line in lines:
        if not isinstance(line, sv.Line):
            error_msg = f"Expected Line, got {type(line)}."
            local_logger.error(error_msg)
            raise TypeError(error_msg)

        sns.lineplot(
            x=[line.left_bottom[0], line.right_top[0]],
            y=[line.left_bottom[1], line.right_top[1]],
            color=line.color,
            linestyle=line.linestyle,
            label=line.label,
            ax=ax,
        )


def savefig_and_close(
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = True,
) -> None:
    """
    Save figure and close it.

    Args:
        filename (str): Filename
        output_dir (str): Output directory
        close (bool): Close figure

    Returns:
        None
    """

    if output_dir:
        utils.check_and_create_dir(dirpath=output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w", bbox_inches="tight")
        local_logger.info("Saved figure to %s.", savepath)

    if close:
        plt.close()
        local_logger.info("Closed figure.")
