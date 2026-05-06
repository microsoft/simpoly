"""
copy from allegro https://github.com/mir-group/allegro/blob/22f673c565d148ae8e9394443baa9f5b5716c8e7/allegro/nn/_strided/_channels.py

nothing was changed except the scatter_sum import
"""

import math

import torch
from e3nn import o3

from simpoly.vivace.modules import e3nn_utils


class MakeWeightedChannels(torch.nn.Module):
    weight_numel: int
    multiplicity_out: int
    _num_irreps: int

    def __init__(  # type: ignore
        self,
        irreps_in,
        multiplicity_out: int,
        pad_to_alignment: int = 1,
    ):
        super().__init__()
        assert all(mul == 1 for mul, ir in irreps_in)
        assert multiplicity_out >= 1
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        w_index = sum(
            ([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in)), []
        )  # TODO type looks wrong
        # pad to padded length
        n_pad = int(math.ceil(irreps_in.dim / pad_to_alignment)) * pad_to_alignment - irreps_in.dim
        # use the last weight, what we use doesn't matter much
        w_index += [w_index[-1]] * n_pad
        self._num_irreps = len(irreps_in)
        self.register_buffer("_w_index", torch.as_tensor(w_index, dtype=torch.long))
        # there is
        self.multiplicity_out = multiplicity_out
        self.weight_numel = len(irreps_in) * multiplicity_out

        self.register_buffer("_w_index_expanded", self._w_index.view(1, 1, -1))

    def forward(self, edge_attr, weights):  # type: ignore

        # special case for graphs with isolated atoms
        if edge_attr.size(0) == 0:
            return edge_attr.view(-1, self.multiplicity_out, edge_attr.size(1))

        # weights are [z, u, i]
        # edge_attr are [z, i]
        # i runs over all irreps, which is why the weights need
        # to be indexed in order to go from [num_i] to [i]
        return torch.einsum(
            "zi,zui->zui",
            edge_attr,
            weights.view(
                -1,
                self.multiplicity_out,
                self._num_irreps,
            ).index_select(2, self._w_index),
        )
