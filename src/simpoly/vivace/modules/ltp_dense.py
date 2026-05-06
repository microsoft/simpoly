import math
import typing as ty
from dataclasses import dataclass
from math import ceil

import torch
from e3nn import o3
from torch.autograd.function import Function, FunctionCtx

from .e3nn_utils import from_e3nn_contractor


@dataclass
class _FunctionCtx(FunctionCtx):
    needs_input_grad: ty.Tuple[bool, bool, bool]
    saved_tensors: ty.Any


# function with only the single backward for deployment
class UnweightedTPDense(torch.nn.Module):
    irreps_in1: str
    irreps_in2: str
    irreps_out: str
    instructions: list[tuple[int, int, int]]

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple[int, int, int]],
    ) -> None:

        super().__init__()
        self.irreps_in1 = str(irreps_in1)
        self.irreps_in2 = str(irreps_in2)
        self.irreps_out = str(irreps_out)
        self.instructions = instructions

        dtype = torch.get_default_dtype()
        (
            sparse_cg,
            index_instructions_list,
            input_size_1,
            input_size_2,
            output_size,
        ) = from_e3nn_contractor(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
        )

        # construct a sparse torch matrix based on this
        dense_cg = torch.zeros((input_size_1, input_size_2, output_size), dtype=dtype)
        for index, value in zip(index_instructions_list, sparse_cg):
            dense_cg[index] = value
        dense_cg = dense_cg.permute(2, 0, 1).contiguous()
        self.register_buffer("dense_cg", dense_cg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("zui, zuj, kij -> zuk", x, y, self.dense_cg)


class UnweightedTPUnrollDense(UnweightedTPDense):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return TPDenseFn.apply(x, y, self.dense_cg)  # type: ignore


class TPDenseFn(Function):
    is_traceable = False

    @staticmethod
    def setup_context(ctx: _FunctionCtx, inputs: ty.Tuple[ty.Any, ...], output: ty.Any) -> None:
        x, y, dense_cg = inputs
        if not ctx.needs_input_grad[0]:
            y = torch.tensor([])
        if not ctx.needs_input_grad[1]:
            x = torch.tensor([])
        if not ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            dense_cg = torch.tensor([])
        ctx.save_for_backward(x, y, dense_cg)

    @staticmethod
    def forward(
        x: torch.Tensor,
        y: torch.Tensor,
        dense_cg: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.einsum("zui, zuj, kij -> zuk", x, y, dense_cg)
        return out

    @staticmethod
    def backward(
        ctx: _FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[ty.Optional[torch.Tensor], ...]:  # type: ignore[override]

        x: torch.Tensor
        y: torch.Tensor
        dense_cg: torch.Tensor

        x, y, dense_cg = ctx.saved_tensors
        assert not ctx.needs_input_grad[2]

        grad_x, grad_y = TPDenseBackwardFn.apply(x, y, dense_cg, grad_output)  # type: ignore[no-untyped-call]

        return grad_x, grad_y, None, None, None


class TPDenseBackwardFn(Function):
    is_traceable = False

    @staticmethod
    def forward(
        ctx: _FunctionCtx,
        x: torch.Tensor,
        y: torch.Tensor,
        dense_cg: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> ty.Tuple[torch.Tensor, ...]:

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x, y, dense_cg, grad_output)

        grad_x = torch.einsum("zuk, zuj, kij -> zui", grad_output, y, dense_cg)
        grad_y = torch.einsum("zuk, zui, kij -> zuj", grad_output, x, dense_cg)

        return grad_x, grad_y

    @staticmethod
    def backward(
        ctx: _FunctionCtx, grad_out_x: torch.Tensor, grad_out_y: torch.Tensor
    ) -> tuple[ty.Optional[torch.Tensor], ...]:  # type: ignore[override]
        if not any(ctx.needs_input_grad):
            return (None,) * 4

        # all or nothing: we now assume that we need to compute *all* gradients to reduce 'if' statements

        x: torch.Tensor
        y: torch.Tensor
        dense_cg: torch.Tensor
        grad_output: torch.Tensor

        x, y, dense_cg, grad_output = ctx.saved_tensors

        grad_grad_x = torch.einsum("zuk, zuj, kij -> zui", grad_output, grad_out_y, dense_cg)
        grad_grad_y = torch.einsum("zui, zuk, kij -> zuj", grad_out_x, grad_output, dense_cg)
        grad_grad_output = torch.einsum("zui, zuj, kij -> zuk", x, grad_out_y, dense_cg)
        grad_grad_output += torch.einsum("zui, zuj, kij -> zuk", grad_out_x, y, dense_cg)

        return grad_grad_x, grad_grad_y, None, grad_grad_output


class ManualDotProduct(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_edge_sh: o3.Irreps) -> None:
        super().__init__()

        out_multiplicity = [mul for mul, ir in irreps_in]
        assert len(set(out_multiplicity)) == 1, "all multiplicity should be the same"
        out_mul = out_multiplicity[0]
        edge_sh_multiplicity = [mul for mul, ir in irreps_edge_sh]
        assert len(set(edge_sh_multiplicity)) == 1, "all multiplicity should be the same"
        out_slices = [slice(s.start / out_mul, s.stop / out_mul) for s in irreps_in.slices()]
        out_keys = [str(ir) for _, ir in irreps_in]
        edge_slices = irreps_edge_sh.slices()
        edge_keys = [str(ir) for _, ir in irreps_edge_sh]

        shared_keys = sorted(set(out_keys).intersection(set(edge_keys)))
        out_slices = [out_slices[out_keys.index(key)] for key in shared_keys]
        edge_slices = [edge_slices[edge_keys.index(key)] for key in shared_keys]

        # convert slices to array of indices
        out_indices = tuple([torch.arange(s.start, s.stop, dtype=torch.long) for s in out_slices])
        edge_indices = tuple([torch.arange(s.start, s.stop, dtype=torch.long) for s in edge_slices])
        out_indices_tensor = torch.cat(out_indices)
        edge_indices_tensor = torch.cat(edge_indices)

        self.register_buffer(
            "out_indices",
            out_indices_tensor,
        )
        self.register_buffer(
            "edge_indices",
            edge_indices_tensor,
        )
        self.inner_product_scaling = 1 / math.sqrt(out_indices_tensor.size(0))

    def forward(self, x: torch.Tensor, spherical_harmonics: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()  # [ne, nf, nc]
        if x.shape[-1] != self.out_indices.shape[0]:
            # this step will throw away 1e, 2o and etc
            x = x.index_select(1, self.out_indices)  # [ne, nf', nc]
        if spherical_harmonics.shape[1] == self.edge_indices.shape[0]:
            sph = spherical_harmonics.unsqueeze(-1)  # [ne, nf, 1]
        else:
            sph = spherical_harmonics.index_select(-1, self.edge_indices).unsqueeze(
                -1
            )  # [ne, nf', 1]
        # expand the node equivariant to edge based on receiver
        # dot product between the two
        x = x * sph  # *[ne, nf', nc]
        x = torch.sum(x, dim=1)  # [ne, nc]
        x = x * self.inner_product_scaling
        return x
