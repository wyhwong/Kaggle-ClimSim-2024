from dataclasses import dataclass


@dataclass
class Padding:
    """Padding of the plot."""

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass
class Labels:
    """Labels of the plot."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""


@dataclass
class Line:
    """Line of the plot."""

    left_bottom: tuple[float, float]
    right_top: tuple[float, float]
    color: str = "blue"
    linestyle: str = "--"
    label: str = ""
