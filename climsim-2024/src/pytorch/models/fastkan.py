"""
    This implementation is based on Approximation of Kolmogorov-Arnold Network
    Original implementation: https://github.com/ZiyaoLi/fast-kan

    Here we change it to be capable of training with torch lightning
"""

from typing import Any, Callable, Optional

import lightning
import torch
import torch.nn as nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


class SplineLinear(nn.Linear):
    """Linear layer with spline initialization"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            init_scale: standard deviation of the truncated normal distribution

        Returns:
            None
        """

        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kwargs)

    def reset_parameters(self) -> None:
        """Initialize the weight with a truncated normal distribution"""

        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(lightning.LightningModule):
    """Radial basis function"""

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: Optional[float] = None,  # larger denominators lead to smoother basis
    ):
        """
        Args:
            grid_min: minimum value of the grid
            grid_max: maximum value of the grid
            num_grids: number of grids
            denominator: denominator of the Gaussian function

        Returns:
            None
        """

        super().__init__()

        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """Compute the radial basis function"""

        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(lightning.LightningModule):
    """Fast Kolmogorov-Arnold Network Layer"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=nn.functional.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: size of each input sample
            output_dim: size of each output sample
            grid_min: minimum value of the grid
            grid_max: maximum value of the grid
            num_grids: number of grids
            use_base_update: whether to use base update
            base_activation: activation function for the base update
            spline_weight_init_scale: standard deviation of the truncated normal distribution

        Returns:
            None
        """

        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, time_benchmark=False) -> torch.Tensor:
        """
        Args:
            x: input tensor
            time_benchmark: whether to benchmark the computation time

        Returns:
            output tensor
        """

        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(ModelBase):
    """Fast Kolmogorov-Arnold Network Model"""

    def __init__(
        self,
        layers_hidden: list[int],
        scheduler_config: Optional[dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=nn.functional.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        """Initialize the model.

        Args:
            layers_hidden: list of hidden layer sizes
            scheduler_config: scheduler configuration
            loss_fn: loss function
            grid_min: minimum value of the grid
            grid_max: maximum value of the grid
            num_grids: number of grids
            use_base_update: whether to use base update
            base_activation: activation function for the base update
            spline_weight_init_scale: standard deviation of the truncated normal distribution

        Returns:
            None
        """

        super().__init__(scheduler_config=scheduler_config, loss_fn=loss_fn)

        self._layers = nn.ModuleList(
            [
                FastKANLayer(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""

        for layer in self._layers:
            x = layer(x)
        return x
