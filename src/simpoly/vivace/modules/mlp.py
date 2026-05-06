import math
from typing import Any, Optional, OrderedDict

import torch
import torch.nn as nn

from simpoly.vivace.modules.shifted_soft_plus import ShiftedSoftPlus


class MLP(torch.nn.Sequential):
    """MLP with custom initialization and normalization, bias are initialized to zero"""

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
        zero_bias: bool = False,
        batch_norm: bool = False,
        bias: bool = True,
        use_rms_norm: bool = False,
        backward_compatible: bool = False,
    ):

        dimensions: list[int] = (
            ([input_dim] if input_dim is not None else [])
            + hidden_dims
            + ([output_dim] if output_dim is not None else [])
        )
        if activation_list is None:
            activation_list = [activation] * (len(dimensions) - 2)

        # these numbers are computed on 1 million random numbers from a uniform distribution in [-1, 1]
        # gen = torch.Generator(device="cpu").manual_seed(0)
        # z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
        # f(z).pow(2).mean().pow(-0.5).item()
        nonlin_const_dict: dict[Any, float] = {
            None: 1.0,
            "silu": 1.679176792398942,
            "ssp": 1.878204668541552,
            "sigmoid": 1.8467055342154763,
        }
        nonlin_const_dict["swish"] = nonlin_const_dict["silu"]
        nonlinearities = {
            None: None,
            "silu": torch.nn.SiLU,
            "ssp": ShiftedSoftPlus,
            "swish": torch.nn.SiLU,
            "sigmoid": torch.nn.Sigmoid,
        }
        assert len(dimensions) >= 2  # Must have input and output dim

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        scalar_mlp_norm_const = []
        prev_norm_const = 1.0
        layers: dict[str, torch.nn.Module] = {}
        for idx, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            if dropout_p > 0:
                layers[f"dropout_{idx}"] = torch.nn.AlphaDropout(p=dropout_p)

            # first check whether this is a layer with a nonlinearity
            activation = activation_list[idx] if idx < len(activation_list) else None
            activation = None if activation is None else activation.lower()
            nonlinearity = nonlinearities[activation]

            linear = torch.nn.Linear(in_features=h_in, out_features=h_out, bias=bias)
            linear_str = f"linear: {h_in}x{h_out}"

            linear_str += f" {initialization}"

            w = linear.weight.data
            bound = 1.0 / math.sqrt(float(h_in))

            if use_rms_norm:
                # this part comes from e3nn, where they normalize the weights
                # depending on the following up activation function
                bound = bound * prev_norm_const
                prev_norm_const = nonlin_const_dict[activation]

            if initialization == "normal":
                w.normal_(0, bound)
            elif initialization == "uniform":
                # these values give < x^2 > = 1
                # note, torch default is uniform(-1, 1)
                w.uniform_(-bound * math.sqrt(3.0), bound * math.sqrt(3.0))
            # # TODO: this part is not right atm
            elif initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(f"Invalid mlp_initialization {initialization}")

            scalar_mlp_norm_const.append(bound)
            linear_str += f" norm_const: {bound:.3f}"

            if bias:
                if zero_bias:
                    linear.bias.data.zero_()
                    linear_str += " (bias zeroed)"
                else:
                    linear.bias.data.uniform_(-bound, bound)
                    linear_str += f" (bias bounded +-{bound:.3f})"

            layers[f"linear_{idx}"] = linear

            if batch_norm:
                layers[f"batch_norm_{idx}"] = torch.nn.BatchNorm1d(h_out)

            if nonlinearity is not None:
                layers[f"act_{idx}"] = nonlinearity()

        if dropout_p > 0:
            # with normal dropout everything blows up
            layers["final_dorpout"] = torch.nn.AlphaDropout(p=dropout_p)

        super().__init__(OrderedDict(layers))

        if backward_compatible:
            self.register_buffer(
                "norm_const",
                torch.as_tensor(scalar_mlp_norm_const, dtype=torch.get_default_dtype()),
            )
