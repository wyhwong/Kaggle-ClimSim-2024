from typing import Callable, Optional

import torch
import torch.nn as nn

import src.logger
from src.pytorch.models.base import ModelBase


local_logger = src.logger.get_logger(__name__)


class Transformer(ModelBase):
    """Sequence to sequence transformer model for regression."""

    def __init__(
        self,
        steps_per_epoch: int,
        input_dim: int = 556,
        output_dim: int = 368,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize the model."""

        super().__init__(steps_per_epoch=steps_per_epoch, loss_fn=loss_fn or nn.CrossEntropyLoss())

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout

        self._embedding_layer = nn.Linear(input_dim, hidden_dim)
        self._transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self._fc_layer = nn.Linear(hidden_dim, output_dim)

        super().__post_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        x = self._embedding_layer(x)
        x = self._transformer(x, x)
        x = self._fc_layer(x)
        return x
