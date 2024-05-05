from dataclasses import dataclass

import numpy as np


@dataclass
class Domain:
    """Domain schema"""

    left: float = -np.inf
    right: float = np.inf

    def __post_init__(self) -> None:
        """Check if minimum is smaller than maximum"""

        if self.left > self.right:
            raise ValueError("Minimum cannot be larger than maximum")

    def __str__(self) -> str:
        """Return string representation"""

        return f"[{self.left}, {self.right}]"
