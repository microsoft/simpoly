import torch
from torch import Tensor

from simpoly.vivace.utils import scatter


def softmax(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_size: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes a sparsely evaluated softmax."""
    # Note: require explicit dim_size; the prior `int(index.max())+1`
    # fallback was a CPU↔GPU sync that blocked CUDAGraph capture.
    assert dim_size is not None, "softmax requires explicit dim_size ( sync removal)."
    size = list(src.size())
    size[dim] = dim_size

    src_max = torch.zeros(size, dtype=src.dtype, device=src.device)
    index_ = scatter.broadcast(index, src, dim)
    src_max.scatter_reduce_(dim, index_, src, reduce="amax", include_self=False)

    src = src - src_max.index_select(dim, index)

    denom_sum = scatter.scatter_sum(src=src.exp(), index=index, dim=dim, dim_size=dim_size) + eps
    denom_sum = denom_sum.index_select(dim, index)
    src -= denom_sum.log()
    return src.exp()


# @torch.jit.script
def softmax_cutoff(
    src: Tensor,
    cutoff: Tensor,
    index: Tensor,
    dim: int,
    dim_size: int | None = None,
    eps: float = 1.0,
) -> torch.Tensor:
    """Computes a sparsely evaluated softmax.
    Cutoff value allows some terms to continuously fade out from the softmax
    - Denominator contains exp(value)*cutoff
    - Nominator contains exp(value)*cutoff

    NOTE: default of eps is 1.0 for eps because cutoff will bring in 0.0 entries.
    In some cases, all entries are zeros, so we need to add a value to ensure the
    yielded value is not NaN. But if eps is too small, this still lead to a sudden
    increase of the value. So we choose 1.0 as default.
    """

    # Make sure the input shapes are correct
    assert src.shape[0] == cutoff.shape[0]
    extra_dims = src.dim() - cutoff.dim()
    for _ in range(extra_dims):
        cutoff = cutoff.unsqueeze(-1)

    # Note: thread `dim_size` from the caller; the previous
    # `int(index.max())+1` was a CPU↔GPU sync that blocked CUDAGraph capture.
    # The `src.numel() == 0` early-out is also data-dependent. We keep it
    # behind a `torch._check` so the export path can resolve the symbolic
    # shape, but in eager its semantics are unchanged.
    assert dim_size is not None, "softmax_cutoff requires explicit dim_size ( sync removal)."

    # Deal with empty tensor scenario
    if src.numel() == 0:
        return torch.empty_like(src)

    size = list(src.size())
    size[dim] = dim_size

    # NOTE: log_cutoff = torch.clamp(log_cutoff, min=-30) would avoid -inf
    # but still be numerically instable and lead to NaNs
    log_cutoff = (torch.abs(cutoff) + 1e-16).log()
    src = src + log_cutoff

    # subtract max for stability
    src_max = torch.zeros(size, dtype=src.dtype, device=src.device)
    index_ = scatter.broadcast(index, src, dim)
    src_max.scatter_reduce_(dim, index_, src, reduce="amax", include_self=False)
    src_max = src_max.index_select(dim, index)
    src = src - src_max  # cannot use inplace operation here due to jit script

    src_exp = src.exp()

    # watch out, when eps is too small, this will lead to NaN or bring in discontinuity
    denom_sum = scatter.scatter_sum(src=src_exp, index=index, dim=dim, dim_size=dim_size) + eps
    denom_sum = denom_sum.index_select(dim, index)

    src -= denom_sum.log()
    return src.exp()
