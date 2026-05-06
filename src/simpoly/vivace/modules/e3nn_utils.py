from typing import List, Tuple, TypeAlias

import torch
from e3nn import o3

from simpoly.vivace.utils import scatter

from .clebsch_gorden import clebsch_gordan_from_sympy


def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps,
    irreps2: o3.Irreps,
    target_irreps: o3.Irreps,
) -> Tuple[o3.Irreps, List[Tuple[int, int, int, str, bool]]]:
    r"""Obtain output irreps of tensor product in a format that it is sorted by \ell."""
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions: List[Tuple[int, int, int, str, bool]] = []

    for i, (mul, irrep1) in enumerate(irreps1):
        for j, (_, irrep2) in enumerate(irreps2):
            for irrep_out in irrep1 * irrep2:  # | l1 - l2 | <= l <= l1 + l2
                if irrep_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, irrep_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product by \ell so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permutation, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permutation[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


class IrrepsReshaper(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()

        # This operation does not make much sense if the irreps are empty
        assert len(irreps) > 0

        mul0 = irreps[0][0]
        assert isinstance(mul0, int)
        self.n_channels = mul0

        assert all(
            mul == self.n_channels for mul, _irrep in irreps
        ), f"All `mul` (or, number of channels) need to be equal in '{irreps}'"

        self.dims = list(irrep.dim for _mul, irrep in irreps)

    def forward(
        self,
        tensor: torch.Tensor,  # [batch, irreps]
    ) -> torch.Tensor:  # [batch, channels, dims]
        assert tensor.dim() == 2
        n_batch = tensor.shape[0]
        blocks = []
        index = 0
        for dim in self.dims:
            width = self.n_channels * dim
            block = tensor[:, index : index + width]  # [batch, mul * n_dim]
            block = block.reshape(n_batch, self.n_channels, dim)
            index += width
            blocks.append(block)

        return torch.cat(blocks, dim=-1)  # [batch, channels, dims]


def extract_num_channels(irreps: o3.Irreps) -> list[int]:
    return list(n for (n, (_ell, _p)) in irreps)


def extract_scalar_irreps(irreps: o3.Irreps) -> o3.Irreps:
    return o3.Irreps([(mul, (ell, p)) for (mul, (ell, p)) in irreps if ell == 0])


def reach_through_linear_transforms(irreps: o3.Irreps, target: o3.Irreps) -> o3.Irreps:
    return o3.Irreps((n, s) for n, s in target if s in irreps)


def from_e3nn_contractor(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, list[tuple[int, int, int]], int, int, int]:
    """This code convert the e3nn input to the index.
    It assume that each possible (l, m, p) has the same multiplicity and they are continuous
    Meaning the input matrix will be (n, n_multiplicity, ((2l_1)+1, (2l_2)+1, (2l_3)+1, ...))
    """

    assert all(mul == irreps_in1[0].mul for mul, _ in irreps_in1)
    assert all(mul == irreps_in2[0].mul for mul, _ in irreps_in2)
    assert all(mul == irreps_out[0].mul for mul, _ in irreps_out)

    instructions_lp = e3nn_to_instructions_lp(irreps_in1, irreps_in2, irreps_out, instructions)
    cg_param, cg_index_lmp = clebsch_gordan_from_sympy(
        dtype=torch.get_default_dtype(),
        instructions=instructions_lp,
    )

    l1_m1_p1 = [
        (irrep.ir.l, m, irrep.ir.p)
        for irrep in irreps_in1
        for m in range(-irrep.ir.l, irrep.ir.l + 1)
    ]
    l2_m2_p2 = [
        (irrep.ir.l, m, irrep.ir.p)
        for irrep in irreps_in2
        for m in range(-irrep.ir.l, irrep.ir.l + 1)
    ]
    l3_m3_p3 = [
        (irrep.ir.l, m, irrep.ir.p)
        for irrep in irreps_out
        for m in range(-irrep.ir.l, irrep.ir.l + 1)
    ]
    input_size_1 = len(l1_m1_p1)
    input_size_2 = len(l2_m2_p2)
    output_size = len(l3_m3_p3)  # L3^2 if parity_invariant

    # translate instruction back to index
    index_instructions_list: list[tuple[int, int, int]] = []
    for l1, m1, p1, l2, m2, p2, l3, m3, p3 in cg_index_lmp:
        index_instructions_list.append(
            (
                l1_m1_p1.index((l1, m1, p1)),
                l2_m2_p2.index((l2, m2, p2)),
                l3_m3_p3.index((l3, m3, p3)),
            )
        )
    return (
        cg_param,
        index_instructions_list,
        input_size_1,
        input_size_2,
        output_size,
    )


def get_slices(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, int]:

    assert all(mul == irreps_in1[0].mul for mul, _ in irreps_in1)
    assert all(mul == irreps_in2[0].mul for mul, _ in irreps_in2)
    assert all(mul == irreps_out[0].mul for mul, _ in irreps_out)

    instructions_lp = e3nn_to_instructions_lp(irreps_in1, irreps_in2, irreps_out, instructions)
    l3_m3_p3 = [
        (irrep.ir.l, m, irrep.ir.p)
        for irrep in irreps_out
        for m in range(-irrep.ir.l, irrep.ir.l + 1)
    ]
    output_size = len(l3_m3_p3)  # L3^2 if parity_invariant

    l1_p1_slice = irreps_to_lp_slice(irreps_in1)
    l2_p2_slice = irreps_to_lp_slice(irreps_in2)
    l3_p3_slice = irreps_to_lp_slice(irreps_out)
    instructions_lp = e3nn_to_instructions_lp(irreps_in1, irreps_in2, irreps_out, instructions)

    # construct a sparse torch matrix based on this
    slices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for l1, p1, l2, p2, l3, p3 in instructions_lp:
        slices.append((l1_p1_slice[(l1, p1)], l2_p2_slice[(l2, p2)], l3_p3_slice[(l3, p3)]))
    slice_tensor = torch.tensor(slices, dtype=torch.long, requires_grad=False)
    return slice_tensor, output_size


def get_dense_cg_block_slices(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: list[tuple[int, int, int]],
) -> tuple[list[torch.Tensor], torch.Tensor, int]:

    assert all(mul == irreps_in1[0].mul for mul, _ in irreps_in1)
    assert all(mul == irreps_in2[0].mul for mul, _ in irreps_in2)
    assert all(mul == irreps_out[0].mul for mul, _ in irreps_out)

    instructions_lp = e3nn_to_instructions_lp(irreps_in1, irreps_in2, irreps_out, instructions)
    sparse_cg_values, cg_index_lmp = clebsch_gordan_from_sympy(
        dtype=torch.get_default_dtype(),
        instructions=instructions_lp,
    )

    l1_p1_slice = irreps_to_lp_slice(irreps_in1)
    l2_p2_slice = irreps_to_lp_slice(irreps_in2)
    l3_p3_slice = irreps_to_lp_slice(irreps_out)
    instructions_lp = e3nn_to_instructions_lp(irreps_in1, irreps_in2, irreps_out, instructions)

    # construct a sparse torch matrix based on this
    dense_cg_blocks_dict: dict[tuple[int, int, int, int, int, int], torch.Tensor] = {}
    slices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for l1, p1, l2, p2, l3, p3 in instructions_lp:
        dense_cg_blocks_dict[(l1, p1, l2, p2, l3, p3)] = torch.zeros(
            (2 * l1 + 1),
            (2 * l2 + 1),
            (2 * l3 + 1),
            dtype=sparse_cg_values.dtype,
            requires_grad=False,
        )
        slices.append((l1_p1_slice[(l1, p1)], l2_p2_slice[(l2, p2)], l3_p3_slice[(l3, p3)]))

    for (l1, m1, p1, l2, m2, p2, l3, m3, p3), value in zip(cg_index_lmp, sparse_cg_values):
        dense_cg_blocks_dict[(l1, p1, l2, p2, l3, p3)][m1 + l1, m2 + l2, m3 + l3] = value

    l3_m3_p3 = [
        (irrep.ir.l, m, irrep.ir.p)
        for irrep in irreps_out
        for m in range(-irrep.ir.l, irrep.ir.l + 1)
    ]
    output_size = len(l3_m3_p3)  # L3^2 if parity_invariant
    slice_tensor = torch.tensor(slices, dtype=torch.long, requires_grad=False)
    return list(dense_cg_blocks_dict.values()), slice_tensor, output_size


def irreps_to_lp_slice(irreps: o3.Irreps) -> dict[tuple[int, int], tuple[int, int]]:
    l_p_slice: dict[tuple[int, int], tuple[int, int]] = {}
    idx = 0
    for irrep in irreps:
        l = irrep.ir.l
        p = irrep.ir.p
        l_p_slice[(l, p)] = (idx, idx + (2 * l + 1))
        idx += 2 * l + 1
    return l_p_slice


def e3nn_to_instructions_lp(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int, int, int]]:
    instructions_lp = []
    for idx1, idx2, idx3 in instructions:
        irrep1 = irreps_in1[idx1].ir
        irrep2 = irreps_in2[idx2].ir
        irrep3 = irreps_out[idx3].ir
        instructions_lp.append(
            (
                irrep1.l,
                irrep1.p,
                irrep2.l,
                irrep2.p,
                irrep3.l,
                irrep3.p,
            )
        )
    return instructions_lp


DeviceLikeType: TypeAlias = str | torch.device | int


def get_irreps_dimensions(
    irreps: o3.Irreps,
    device: DeviceLikeType | None = None,
) -> torch.Tensor:
    """
    Get the dimensions of the irreps in the flattened representation.
    Example: 1x0e + 2x1o + 1x0e -> [1, 3, 3, 1]
    """
    dims = [dim for mul, irrep in irreps for dim in [irrep.dim] * mul]
    return torch.tensor(dims, dtype=torch.long, device=device)


def get_irreps_indices(
    irreps: o3.Irreps,
    device: DeviceLikeType | None = None,
) -> torch.Tensor:
    """
    Get the indices of the irreps in the flattened representation.
    Example: 1x0e + 2x1o + 1x0e -> [0, 1, 1, 1, 2, 2, 2, 3]
    """
    indices = []
    flat_irreps: list[o3.Irrep] = [ir for mul, ir in irreps for _ in range(mul)]  # [0e, 1o, 1o, 0e]
    for i, irrep in enumerate(flat_irreps):
        indices.extend([i] * irrep.dim)
    return torch.tensor(indices, dtype=torch.long, device=device)


class ComputeNorms(torch.nn.Module):
    """
    Reduces each irrep to its norm. So the number of features is compressed to the number of irreps.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        squared: bool = False,
        eps: float = 1e-8,
        device: DeviceLikeType | None = None,
    ):
        super().__init__()

        self.irreps_in = irreps
        self.irreps_out = o3.Irreps((mul, o3.Irrep("0e")) for mul, _ in self.irreps_in)

        # self.irreps_out is no longer an e3nn object after jit scripting
        # Therefore, we store self.irreps_out.dim here and use that in the forward
        self.irreps_out_dim = self.irreps_out.dim

        self.register_buffer("_indices", get_irreps_indices(self.irreps_in, device))
        self.squared = squared
        self.eps = eps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def forward(
        self,
        x: torch.Tensor,  # [..., n_feats]
    ) -> torch.Tensor:
        squared = torch.square(x)
        summed = scatter.scatter_sum(
            src=squared,
            index=self._indices,
            dim=-1,  # scatter along the last (feat) dimension
            dim_size=self.irreps_out_dim,
        )

        if not self.squared:
            summed = torch.sqrt(summed + self.eps**2) - self.eps

        return summed  # [..., n_out_feats]


def get_invariant_indices(irreps: o3.Irreps, device: DeviceLikeType | None = None) -> torch.Tensor:
    """
    Get the indices of the 0e irreps.
    Example: 1x0e + 2x1o + 2x0e -> [0, 4, 5]
    """
    indices: list[torch.Tensor] = []
    pos = 0

    for mul, ir in irreps:
        width = ir.dim * mul
        if ir == o3.Irrep("0e"):
            indices.append(torch.arange(pos, pos + width, dtype=torch.long))
        pos += width

    if len(indices) == 0:
        raise ValueError(f"Irreps '{irreps}' does not contain any 0e irreps")

    return torch.cat(indices, dim=0).to(device)


class SelectInvariants(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, device: DeviceLikeType | None = None) -> None:
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = o3.Irreps(
            (mul, irrep) for (mul, irrep) in irreps if irrep == o3.Irrep("0e")
        )

        self.register_buffer("_indices", get_invariant_indices(self.irreps_in, device))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def forward(
        self,
        x: torch.Tensor,  # [..., n_feats]
    ) -> torch.Tensor:
        return x.index_select(-1, self._indices)
