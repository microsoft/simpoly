import math
from abc import ABC, abstractmethod

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import torch
from e3nn import o3

# NOTE: CuEquivariqnce expects tensors of shape [batch, n_channels * n_irreps]
# Our input always has 3 dimensions, so there are 2 possible scenarios:
# - [batch, n_channels, n_irreps]
# - [batch, n_irreps, n_channels]

# Force the FX/einsum fallback for every cuet.TensorProduct in this module.
#
# cuequivariance ships two backends:
#   - FusedTensorProductOp3/4 — fused CUDA kernels (selected by
#     ``use_fallback=False`` or ``None`` when CUDA is available)
#   - torch.fx GraphModule of einsum chains (selected by ``use_fallback=True``)
#
# For our LAMMPS workloads (typical 200 – 10k atoms / step) the fused CUDA
# path is *slower* than the einsum fallback: the per-call dispatch overhead
# of FusedTensorProductOp3 plus the dynamic-shape recompiles of the outer
# torch.compile wrapper dominate at this scale, costing ~25 % on pp_7398.
# The fused kernel only wins on much larger irreps / batches than we run.
# See campaigns/2026-05-vivace-cueq-padded-wrapper/REPORT.md and the
# checkpoint regen note in internal/checkpoint_conversion/.
_USE_CUEQ_FALLBACK = True


def get_multiplicities(irreps: o3.Irreps) -> set[int]:
    return set([mul for mul, _ in irreps])


# -----------------
# Tensor Product
# -----------------
class LTPCuEquivarianceBase(torch.nn.Module, ABC):
    irreps_in1: str
    irreps_in2: str
    irreps_out: str
    instructions: list[tuple[int, int, int]]
    n_channels: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.n_channels = self._get_multiplicity(irreps_in1, irreps_in2, irreps_out)
        self.tp = self._build_tp(irreps_in1, irreps_in2, irreps_out, instructions)
        self.irreps_in1 = str(irreps_in1)
        self.irreps_in2 = str(irreps_in2)
        self.irreps_out = str(irreps_out)
        self.instructions = instructions

    def _get_multiplicity(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
    ) -> int:
        stub = "must have the same multiplicity as this is uuu tensor product"
        ms = {}
        for _n, _i in zip(["in1", "in2", "out"], [irreps_in1, irreps_in2, irreps_out]):
            ms[_n] = get_multiplicities(_i)
            assert len(ms[_n]) == 1, f"All irreps_{_n} {stub}"
        assert ms["in1"] == ms["in2"], "Irreps in1 and in2 must have the same multiplicity"
        assert ms["in1"] == ms["out"], "Irreps in1 and out must have the same multiplicity"
        return ms["in1"].pop()

    def _build_tp(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple[int, int, int]],
    ) -> torch.nn.Module:

        stp = cue.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
        old_irreps_list: list[o3.Irreps] = [irreps_in1, irreps_in2, irreps_out]
        irreps_str_list: list[str] = []
        for old_irreps in old_irreps_list:
            irreps_str = []
            for mul, ir in old_irreps:
                irreps_str.append(str(ir))
            irreps_str_list.append("+".join(irreps_str))
        new_irreps1 = self.n_channels * cue.Irreps(cue.O3, irreps_str_list[0])
        new_irreps2 = self.n_channels * cue.Irreps(cue.O3, irreps_str_list[1])
        new_irreps3 = self.n_channels * cue.Irreps(cue.O3, irreps_str_list[2])

        for i, _irrep in enumerate([new_irreps1, new_irreps2, new_irreps3]):
            for mul, ir in _irrep:
                stp.add_segment(i, (ir.dim, mul))

        for i1, i2, i3 in instructions:
            mul, ir1 = new_irreps1[i1]
            mul, ir2 = new_irreps2[i2]
            mul, ir3 = new_irreps3[i3]
            if ir3 not in ir1 * ir2:
                continue
            # for loop over the different solutions of the Clebsch-Gordan decomposition
            normalization = math.sqrt(1 / sum(1 for ins in instructions if ins[2] == i3))
            for cg in cue.O3.clebsch_gordan(ir1, ir2, ir3):
                stp.add_path(i1, i2, i3, c=cg * normalization)
        return cuet.TensorProduct(  # type: ignore[no-any-return]
            stp.flatten_coefficient_modes(),
            device="cpu",
            use_fallback=_USE_CUEQ_FALLBACK,
            math_dtype=torch.get_default_dtype(),
        )

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


class LTPCuEquivarianceTwist(LTPCuEquivarianceBase):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [batch, irreps_in1, n_channels]
        # y: [batch, irreps_in2, n_channels]
        # tp_out: [batch, irreps_out, n_channels]
        tp_out: torch.Tensor = self.tp(x.view(x.size(0), -1), y.view(y.size(0), -1))
        return tp_out.view(x.size(0), -1, self.n_channels)


# -----------------
# Dot Product
# -----------------
class DotProductCuEquivarianceBase(torch.nn.Module, ABC):
    irreps_in1: str
    irreps_in2: str
    n_channels: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,  # spherical harmonics irreps, must have multiplicity 1
    ) -> None:
        super().__init__()
        self.irreps_in1 = str(irreps_in1)
        self.irreps_in2 = str(irreps_in2)
        self.n_channels = self._get_multiplicity(irreps_in1, irreps_in2)
        self.tp = self._build_tp(irreps_in1, irreps_in2)

    def _get_multiplicity(self, irreps_in1: o3.Irreps, irreps_in2: o3.Irreps) -> int:
        ms1 = get_multiplicities(irreps_in1)
        ms2 = get_multiplicities(irreps_in2)
        assert len(ms1) == 1, "All irreps in irreps_in1 must have the same multiplicity"
        assert len(ms2) == 1, "All irreps in irreps_in2 must have the same multiplicity"
        assert (
            ms2.pop() == 1
        ), "Irreps in irreps_in2 must have multiplicity 1 because it is spherical harmonics"
        return ms1.pop()

    def _build_tp(self, irreps_in1: o3.Irreps, irreps_in2: o3.Irreps) -> torch.nn.Module:
        stp = cue.SegmentedTensorProduct.from_subscripts("iu,j,u+ij")

        # [n_channel, n_irreps_in1]
        for _, ir in irreps_in1:
            stp.add_segment(0, (ir.dim, self.n_channels))
        # [n_irreps_in2]
        for _, ir in irreps_in2:
            stp.add_segment(1, (ir.dim,))
        # [n_channels, 1]
        stp.add_segment(2, (self.n_channels,))

        n_counts = 0
        for i1, (_, ir1) in enumerate(irreps_in1):
            for i2, (_, ir2) in enumerate(irreps_in2):
                if ir1 == ir2:
                    n_counts += ir1.dim
        inner_product_scaling = 1 / math.sqrt(n_counts)

        for i1, (_, ir1) in enumerate(irreps_in1):
            for i2, (_, ir2) in enumerate(irreps_in2):
                if ir1 == ir2:
                    stp.add_path(i1, i2, 0, c=np.eye(ir1.dim) * inner_product_scaling)

        return cuet.TensorProduct(  # type: ignore[no-any-return]
            stp.flatten_coefficient_modes(),
            device="cpu",
            use_fallback=_USE_CUEQ_FALLBACK,
            math_dtype=torch.get_default_dtype(),
        )

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


class DotProductCuEquivarianceTwist(DotProductCuEquivarianceBase):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [batch, irreps_in1, n_channels]
        # y: [batch, irreps_in2]
        tp_out: torch.Tensor = self.tp(x.view(x.size(0), -1), y)  # [batch, n_channels]
        return tp_out


class LinearWrapperBase(torch.nn.Module):
    irreps_in: str
    irreps_out: str
    n_channels_out: int
    w: torch.Tensor
    dtype: str

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.irreps_in = str(irreps_in)
        self.irreps_out = str(irreps_out)
        self.dtype = str(dtype).replace("torch.", "")  # Convert to string without torch prefix
        self.n_channels_out, self.linear, w = self._build_tp(irreps_in, irreps_out, dtype=dtype)
        self.register_parameter("w", torch.nn.Parameter(w))

    def _build_tp(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        dtype: torch.dtype,
    ) -> tuple[int, torch.nn.Module, torch.Tensor]:
        msin = get_multiplicities(irreps_in)
        msout = get_multiplicities(irreps_out)
        assert len(msin) == 1, "All irreps in irreps_in must have the same multiplicity"
        assert len(msout) == 1, "All irreps in irreps_out must have the same multiplicity"
        n_channels_in = msin.pop()
        n_channels_out = msout.pop()

        new_irreps1 = n_channels_in * cue.Irreps(cue.O3, "+".join(str(ir) for _, ir in irreps_in))
        new_irreps2 = n_channels_out * cue.Irreps(cue.O3, "+".join(str(ir) for _, ir in irreps_out))

        d = cue.SegmentedTensorProduct.from_subscripts("vu,iu,iv")
        for _, ir in irreps_in:
            d.add_segment(1, (ir.dim, n_channels_in))

        for _, ir in irreps_out:
            d.add_segment(2, (ir.dim, n_channels_out))

        for i1, (_, ir1) in enumerate(new_irreps1):
            for i2, (_, ir2) in enumerate(new_irreps2):
                if ir1 == ir2:
                    d.add_path(None, i1, i2, c=1.0 / math.sqrt(n_channels_in))
        linear = cuet.TensorProduct(d, use_fallback=_USE_CUEQ_FALLBACK, math_dtype=dtype)
        w = torch.rand(1, d.operands[0].size, dtype=dtype)
        return n_channels_out, linear, w

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class LinearWrapperTwist(LinearWrapperBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, irreps_in1, n_channels]
        # w (n_batch, n_blocks * n_channels_out)
        x = self.linear(self.w, x.view(x.size(0), -1))
        return x.view(x.size(0), -1, self.n_channels_out)


# -----------------
# Make Weighted Channels
# -----------------
class MakeWeightedChannelsBase(torch.nn.Module):
    irreps_in: str
    multiplicity_out: int
    n_channels: int
    n_blocks: int

    def __init__(
        self,
        irreps_in: o3.Irreps,  # spherical harmonics irreps, must have multiplicity 1
        multiplicity_out: int,
    ) -> None:
        super().__init__()
        self.irreps_in = str(irreps_in)
        self.multiplicity_out = multiplicity_out
        self.n_channels, self.n_blocks, self.linear = self._build_tp(irreps_in, multiplicity_out)

    def _build_tp(
        self,
        irreps_in: o3.Irreps,
        multiplicity_out: int,
    ) -> tuple[int, int, torch.nn.Module]:
        msin = get_multiplicities(irreps_in)
        assert len(msin) == 1, "All irreps in irreps_in must have the same multiplicity"
        assert (
            msin.pop() == 1
        ), "Irreps in irreps_in must have multiplicity 1 because it is spherical harmonics"
        n_channels = multiplicity_out
        n_blocks = len(irreps_in)  # Number of irreps in the input

        d = cue.SegmentedTensorProduct.from_subscripts("i,u,iu")
        for _, ir in irreps_in:
            d.add_segment(0, (ir.dim,))
            d.add_segment(1, (n_channels,))
            d.add_segment(2, (ir.dim, n_channels))

        for i1 in range(n_blocks):
            d.add_path(i1, i1, i1, c=1.0)

        linear = cuet.TensorProduct(
            d, use_fallback=_USE_CUEQ_FALLBACK, math_dtype=torch.get_default_dtype()
        )

        return n_channels, n_blocks, linear

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


class MakeWeightedChannelsTwist(MakeWeightedChannelsBase):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # w (n_batch, n_blocks * n_channels_out)
        # x (n_batch, n_irreps)
        x = self.linear(x.view(x.size(0), -1), w)
        return x.view(x.size(0), -1, self.n_channels)
