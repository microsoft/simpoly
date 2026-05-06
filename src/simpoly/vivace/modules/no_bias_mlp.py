from typing import Optional

import torch
from e3nn import o3

from simpoly.vivace import keys
from simpoly.vivace.modules.mlp import MLP


class NoBiasMLP(MLP):
    """Apply an MLP to a dictionary"""

    in_features: int
    out_features: int

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: Optional[int] = None,
        activation: Optional[str] = "silu",
        activation_list: Optional[list[Optional[str]]] = None,
        initialization: str = "uniform",
        dropout_p: float = 0.0,
        batch_norm: bool = False,
        use_rms_norm: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            activation_list=activation_list,
            initialization=initialization,
            dropout_p=dropout_p,
            bias=False,
            batch_norm=batch_norm,
            use_rms_norm=use_rms_norm,
        )
