import math

import torch


def step_clamp(
    x: torch.Tensor,
    r_max_inv: torch.Tensor,
    one_minus_fractional_offset: float,
    fractional_offset_inv: float,
) -> torch.Tensor:
    """
    a cutoff function that stay flat till the last offset and then smoothly decrease to 0
    r_max_inv = 1.0 / r_max
    one_minus_fractional_offset = 1 - offset / r_max
    fractional_offset_inv = r_max / offset
    """
    x = x * r_max_inv
    ones: torch.Tensor = torch.ones_like(x)
    # tmp = (1 - x) * fractional_offset_inv
    tmp = torch.add(
        input=torch.as_tensor(fractional_offset_inv), other=x, alpha=-fractional_offset_inv
    )
    # linking = 3.0 - 2 * tmp
    linking = torch.add(input=torch.as_tensor(3.0), other=tmp, alpha=-2.0)
    tmp **= 2
    linking *= tmp
    # Note: arithmetic `torch.where` instead of data-dependent boolean-mask
    # LHS assignment `out[mask] = linking[mask]`. The boolean-mask form is
    # CUDAGraph-incompatible (count of true entries is known only at runtime,
    # so graph capture aborts with "operation not permitted when stream is
    # capturing"). `torch.where` is bit-identical for fp32 (same predicate,
    # same source tensors, element-wise selection) and is CUDAGraph-safe.
    mask = x > one_minus_fractional_offset
    out: torch.Tensor = torch.where(mask, linking, ones)
    out *= x < 1.0
    return out


def polynomial_clamp_p(x: torch.Tensor, r_max_inv: torch.Tensor, p: int) -> torch.Tensor:
    """
    proposed in DimeNet: https://arxiv.org/abs/2003.03123
    # if r_max_inv is a tensor of shape (n), output will be of shape (n, x.shape)
    """
    assert p >= 2.0
    r_max_inv, x = torch.broadcast_tensors(r_max_inv.unsqueeze(-1), x.unsqueeze(0))  # type: ignore[no-untyped-call]
    x = x * r_max_inv
    x_pow_p = torch.pow(x, p)

    out: torch.Tensor = torch.add(
        torch.as_tensor(1.0), alpha=-((p + 1.0) * (p + 2.0) / 2.0), other=x_pow_p
    )
    out += p * (p + 2.0) * x * x_pow_p
    out -= (p * (p + 1.0) / 2) * x * x * x_pow_p
    out *= x < 1.0

    return out


class StepClamp(torch.nn.Module):
    one_minus_fractional_offset: float
    fractional_offset_inv: float

    def __init__(self, r_max: float, offset: float = 1.0) -> None:

        super().__init__()
        _r_max = torch.as_tensor(r_max, dtype=torch.get_default_dtype())
        _r_max.requires_grad_(False)
        self.register_buffer("r_max_inv", 1.0 / _r_max)

        self.one_minus_fractional_offset = 1 - offset / r_max
        self.fractional_offset_inv = r_max / offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return step_clamp(  # type: ignore[no-any-return]
            x, self.r_max_inv, self.one_minus_fractional_offset, self.fractional_offset_inv
        )


class PolynomialClamp(torch.nn.Module):
    p: int
    r_max: float

    def __init__(self, r_max: float, p: int = 6):

        super().__init__()

        assert p >= 2
        _r_max = torch.as_tensor(r_max, dtype=torch.get_default_dtype())
        _r_max.requires_grad_(False)

        self.register_buffer("r_max_inv", 1.0 / _r_max)
        self.p = p
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # special case for graphs with isolated atoms
        if x.size(0) == 0:
            return x.squeeze(0)
        return polynomial_clamp_p(x, self.r_max_inv, self.p).squeeze(0)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


class CosineClamp(torch.nn.Module):
    """Cosine cutoff function for distance-based features."""

    def __init__(self, r_min: float = 0.0, r_max: float = 5.0) -> None:
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.delta_r = r_max - r_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clamping = x < self.r_max
        if self.r_min > 0:
            lower_clamping = x > self.r_min
            clamping = clamping * lower_clamping
            # cutoffs = 0.5 * (torch.cos(math.pi* (2 * (x - self.r_min) / (self.delta_r) + 1.0))+ 1.0)
            x = torch.add(
                input=torch.as_tensor(-self.r_min / self.delta_r * 2.0),
                other=x,
                alpha=2.0 / self.delta_r,
            )
            x = torch.add(input=torch.as_tensor(math.pi), other=x, alpha=math.pi)
        else:
            # cutoffs = 0.5 * (torch.cos(x * math.pi / self.r_max) + 1.0)
            x = x * (math.pi / self.r_max)
        x = torch.cos(x)
        x = x + 1.0
        x = x * 0.5
        # remove contributions beyond the cutoff radius
        x = x * clamping.float()
        return x
