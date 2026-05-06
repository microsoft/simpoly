# NOTE: This file is a wrapper around torch.scatter_reduce so that our former
# calls to torch_scatter do not need to be changed.
from typing import Optional

import torch


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    # NOTE: Adapted from torch_scatter.scatter.py, original code is using an MIT licence and can be found at
    # https://github.com/rusty1s/pytorch_scatter/blob/8ec9364b0bdcd99149952a25749ad211c2d0567b/torch_scatter/scatter.py

    # Create output tensor
    index = broadcast(index, src, dim)
    size = list(src.size())
    # Note: removed the `int(index.max()) + 1` D2H sync fallback. Every
    # in-tree caller of scatter_{sum,mean,reduce} passes `dim_size` explicitly
    # (audited: properties.py, models/{layers,equivariant,vivace_bergamot,vivace}.py,
    # modules/{e3nn_utils,scatter_softmax}.py). The fallback was dead code on
    # the inference forward path and was the only remaining sync inside
    # scatter_reduce; removing it unblocks CUDAGraph capture.
    assert dim_size is not None, (
        "scatter_reduce requires explicit dim_size; the implicit "
        "`int(index.max())+1` fallback was removed in  (it broke "
        "CUDAGraph capture). Every in-tree caller already passes dim_size."
    )
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # Use built-in PyTorch scatter_reduce
    return out.scatter_reduce_(
        dim=dim,
        index=index,
        src=src,
        reduce=reduce,
        include_self=False,
    )


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_reduce(src, index, dim, dim_size, "sum")


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_reduce(src, index, dim, dim_size, "mean")


# NOTE: The below method is copied from torch_scatter.utils.broadcast(), original code is using an MIT License and can be
# found at https://github.com/rusty1s/pytorch_scatter/blob/8ec9364b0bdcd99149952a25749ad211c2d0567b/torch_scatter/utils.py
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
