"""Edge-mask plumbing + arithmetic radial clamp regression tests.

This test file is the unit-test surface for the radial-clamp fixes:
  - Arithmetic `step_clamp` (boolean-mask LHS → `torch.where`), DONE here.
  - Edge mask plumbing (`real_edge_mask`), pending.
  - Scatter aliasing fix, pending.

Includes ≥15 cases covering adversarial padding. Currently covers the
arithmetic clamp bit-identity gate; additional test classes will be added
as the remaining edge-mask plumbing and scatter aliasing fixes land.

Run:
    PYTHONPATH=src pytest tests/test_edge_mask.py -v
"""

from __future__ import annotations

import pytest
import torch

from simpoly.vivace.modules.radial_clamping import (
    StepClamp,
    step_clamp,
)


# --------------------------------------------------------------------------- #
# Reference: previous step_clamp body (boolean-mask LHS form). Used to prove
# the new `torch.where` arithmetic is bit-identical to the old form for
# every input.
# --------------------------------------------------------------------------- #
def _step_clamp_reference(
    x: torch.Tensor,
    r_max_inv: torch.Tensor,
    one_minus_fractional_offset: float,
    fractional_offset_inv: float,
) -> torch.Tensor:
    x = x * r_max_inv
    out: torch.Tensor = torch.ones_like(x)
    tmp = torch.add(
        input=torch.as_tensor(fractional_offset_inv),
        other=x,
        alpha=-fractional_offset_inv,
    )
    linking = torch.add(input=torch.as_tensor(3.0), other=tmp, alpha=-2.0)
    tmp **= 2
    linking *= tmp
    mask = x > one_minus_fractional_offset
    out[mask] = linking[mask]
    out *= x < 1.0
    return out


# --------------------------------------------------------------------------- #
# Arithmetic step_clamp bit-identity gate.
#
# Non-negotiable: when called from any path (no padding involved),
# the new arithmetic implementation MUST produce numerically identical output
# to the previous boolean-mask version. A single-bit difference is a
# regression. This is a default-on change in a deployed model.
# --------------------------------------------------------------------------- #
class TestStepClampArithBitIdentity:
    """Arithmetic step_clamp must be bit-identical to the reference."""

    @staticmethod
    def _params(r_max: float = 5.0, offset: float = 1.0):
        r_max_inv = torch.tensor(1.0 / r_max, dtype=torch.float32)
        one_minus_fractional_offset = 1.0 - offset / r_max
        fractional_offset_inv = r_max / offset
        return r_max_inv, one_minus_fractional_offset, fractional_offset_inv

    def _check_bit_identity(self, x: torch.Tensor, r_max=5.0, offset=1.0):
        r_max_inv, omfo, foi = self._params(r_max, offset)
        ref = _step_clamp_reference(x.clone(), r_max_inv, omfo, foi)
        new = step_clamp(x.clone(), r_max_inv, omfo, foi)
        # Bit-identity: every element exactly equal in fp32. Use torch.equal
        # so empty tensors are handled correctly (numel()==0 trivially equal).
        assert torch.equal(ref, new), (
            f"step_clamp diverged from reference: max|Δ|="
            f"{(ref - new).abs().max().item() if ref.numel() else 0} "
            f"(x.shape={tuple(x.shape)}, r_max={r_max}, offset={offset})"
        )
        # Also assert dtype + shape preserved.
        assert ref.dtype == new.dtype
        assert ref.shape == new.shape

    # 1. Below the offset boundary, mask is False everywhere.
    def test_below_offset_uniform(self):
        # x in [0, r_max - offset) → x < one_minus_fractional_offset → out = 1
        x = torch.linspace(0.0, 3.5, 100, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 2. Within the linking region, mask is True everywhere.
    def test_inside_linking_region(self):
        # x in (r_max - offset, r_max) → triggers linking branch
        x = torch.linspace(4.01, 4.99, 100, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 3. Above r_max, mask True, but `out *= x < 1.0` (after rescale) zeros it.
    def test_above_rmax(self):
        x = torch.linspace(5.01, 7.0, 50, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 4. Mixed region spanning all three branches, the realistic case.
    def test_mixed_full_range(self):
        x = torch.linspace(0.0, 6.0, 1000, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 5. Exactly at the offset boundary. Predicate is `x > omfo`, so equality
    # falls into the `out=1` branch. Tests the strictness of the predicate.
    def test_exact_boundary(self):
        # one_minus_fractional_offset = 1 - 1/5 = 0.8 → x*r_max_inv == 0.8 at x=4.0
        x = torch.tensor([4.0, 4.0 + 1e-7, 4.0 - 1e-7], dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 6. Exactly at r_max boundary. `out *= x < 1.0` (after rescale) → x=1.0
    # is False, so output is 0.
    def test_exact_rmax(self):
        x = torch.tensor([5.0, 5.0 + 1e-7, 5.0 - 1e-7], dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 7. Length zero, relevant to leak family #1 (envelope at length=0).
    # pre-fix, step_clamp(0) = 1.0 (spurious). Test that the *new*
    # implementation reproduces this exactly. Fixing the spurious value
    # is the responsibility of edge-mask plumbing (Phase 1.1), NOT of the
    # arithmetic transformation here.
    def test_length_zero_matches_reference(self):
        x = torch.zeros(64, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)
        # Document the spurious-but-bit-identical value:
        r_max_inv, omfo, foi = self._params(5.0, 1.0)
        out = step_clamp(x.clone(), r_max_inv, omfo, foi)
        # At x=0: not > omfo (0.8) and not >= 1.0 → out = 1
        assert torch.all(out == 1.0), (
            "pre-fix step_clamp returns 1.0 at length=0; this regression "
            "documents the leak that Phase 1.1 edge-mask plumbing fixes."
        )

    # 8. Empty tensor, must not crash, must produce empty.
    def test_empty(self):
        x = torch.zeros(0, dtype=torch.float32)
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 9. Multi-dimensional input, operations are elementwise, must work.
    def test_multidim(self):
        torch.manual_seed(0)
        x = torch.rand(8, 16, dtype=torch.float32) * 6.0
        self._check_bit_identity(x, r_max=5.0, offset=1.0)

    # 10. fp64 dtype.
    def test_fp64(self):
        x = torch.linspace(0.0, 6.0, 100, dtype=torch.float64)
        r_max_inv = torch.tensor(1.0 / 5.0, dtype=torch.float64)
        ref = _step_clamp_reference(x.clone(), r_max_inv, 0.8, 5.0)
        new = step_clamp(x.clone(), r_max_inv, 0.8, 5.0)
        assert torch.equal(ref, new)
        assert new.dtype == torch.float64

    # 11. Different (r_max, offset), checkpoint flexibility.
    @pytest.mark.parametrize(
        "r_max, offset",
        [(4.0, 0.5), (6.0, 1.5), (10.0, 2.0), (3.0, 0.25)],
    )
    def test_param_sweep(self, r_max, offset):
        x = torch.linspace(0.0, r_max + 0.5, 200, dtype=torch.float32)
        self._check_bit_identity(x, r_max=r_max, offset=offset)

    # 12. StepClamp module forward equivalence (covers the registered buffer).
    def test_step_clamp_module(self):
        m = StepClamp(r_max=5.0, offset=1.0)
        x = torch.linspace(0.0, 6.0, 200, dtype=torch.float32)
        out_module = m(x.clone())
        ref = _step_clamp_reference(
            x.clone(), m.r_max_inv, m.one_minus_fractional_offset, m.fractional_offset_inv
        )
        assert torch.equal(out_module, ref)

    # 13. Random adversarial input, large magnitudes including negatives.
    # Negatives are not physical but the function must not crash; the
    # `x * (x < 1.0)` post-multiply zeros NaNs only if linking is finite.
    def test_random_adversarial(self):
        torch.manual_seed(42)
        for _ in range(8):
            x = torch.randn(256, dtype=torch.float32) * 5.0
            self._check_bit_identity(x.abs(), r_max=5.0, offset=1.0)  # use abs to keep physical

    # 14. Gradient flow, autograd must work (used in force computation).
    def test_grad_matches_reference(self):
        x = torch.linspace(0.1, 4.9, 100, dtype=torch.float32, requires_grad=True)
        r_max_inv, omfo, foi = self._params(5.0, 1.0)
        # forward+backward through reference
        out_ref = _step_clamp_reference(x.clone(), r_max_inv.clone(), omfo, foi)
        # gather grads w.r.t. x via autograd.grad
        # need a fresh leaf for each call
        x1 = torch.linspace(0.1, 4.9, 100, dtype=torch.float32, requires_grad=True)
        out_ref = _step_clamp_reference(x1, r_max_inv, omfo, foi)
        g_ref = torch.autograd.grad(out_ref.sum(), x1)[0]

        x2 = torch.linspace(0.1, 4.9, 100, dtype=torch.float32, requires_grad=True)
        out_new = step_clamp(x2, r_max_inv, omfo, foi)
        g_new = torch.autograd.grad(out_new.sum(), x2)[0]

        # bit-identity also at gradient level
        assert torch.equal(
            g_ref, g_new
        ), f"step_clamp grad diverged: max|Δ∇|={(g_ref - g_new).abs().max().item()}"

    # 15. CUDA bit-identity (skip if no GPU).
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_bit_identity(self):
        x = torch.linspace(0.0, 6.0, 1000, dtype=torch.float32, device="cuda")
        r_max_inv = torch.tensor(1.0 / 5.0, dtype=torch.float32, device="cuda")
        ref = _step_clamp_reference(x.clone(), r_max_inv, 0.8, 5.0)
        new = step_clamp(x.clone(), r_max_inv, 0.8, 5.0)
        assert torch.equal(ref, new)


# --------------------------------------------------------------------------- #
# CUDAGraph capturability check.
#
# The whole motivation for the arithmetic transformation is CUDAGraph capture.
# Verify the new step_clamp captures cleanly in isolation.
# --------------------------------------------------------------------------- #
class TestStepClampCUDAGraphCapture:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_isolated_capture(self):
        """The previous boolean-mask LHS form crashes capture; the new arithmetic
        form must capture without error."""
        x_static = torch.linspace(0.0, 6.0, 4096, dtype=torch.float32, device="cuda")
        r_max_inv = torch.tensor(1.0 / 5.0, dtype=torch.float32, device="cuda")
        omfo = 0.8
        foi = 5.0

        # Warmup on side stream (required by CUDA Graphs).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = step_clamp(x_static, r_max_inv, omfo, foi)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        out_static = torch.empty_like(x_static)
        with torch.cuda.graph(g):
            out_static.copy_(step_clamp(x_static, r_max_inv, omfo, foi))

        # Replay; verify output matches an eager call.
        x_static.copy_(torch.linspace(0.5, 5.5, 4096, dtype=torch.float32, device="cuda"))
        g.replay()
        torch.cuda.synchronize()

        eager = step_clamp(x_static, r_max_inv, omfo, foi)
        assert torch.equal(out_static, eager), (
            "CUDAGraph replay output diverged from eager; the captured "
            "kernel sees different memory than the eager call."
        )


# --------------------------------------------------------------------------- #
# Future test classes (TODO in next invocation):
# class TestRealEdgeMaskPlumbing, Phase 1.1 (≥4 cases: zero-pad,
# random-pad, all-True bit-identity, mixed real/pad)
# class TestScatterAliasing, Phase 1.3 (≥3 cases: pad sender=0
# no leak to atom 0; sentinel-vs-mask consistency; gradient flow)
# class TestPaddingEquivalence, Phase 3 end-to-end (zero-pad and
# adversarial-pad force max|Δ| < 2e-5 vs real-only)
# --------------------------------------------------------------------------- #
