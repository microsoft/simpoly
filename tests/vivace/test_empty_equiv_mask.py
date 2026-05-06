"""Regression test pinning Vivace MLIAP behaviour on stretched systems.

Note:(AOTI exportability) needs to know whether the
``if n_equiv_edges == 0:`` guards in ``vivace.models.layers.InitHeader.forward``
(lines ~278, ~332) and the ``if edge_mask.sum() == 0`` early-returns
``vivace.models.equivariant`` are *load-bearing* (i.e. eager forward crashes
without them) or *defensive* (eager forward returns finite output via the empty
branch).

This test constructs a synthetic batch with ``raw npairs > 0`` but
``edge_length > equiv_r_max`` for *every* edge, so the equivariant edge mask is
all-False (``n_equiv_edges == 0``). Running the eager forward through this
pathological input pins the ground-truth contract: the model must produce
finite forces and a finite scalar energy.

The pinned ground truth is what the AOTI OUTER-fallback
``vivace.mliap.MLFFModelMLIAP`` must reproduce on stretched inputs, *because*
the post- export-friendly inner code path will assume
``n_equiv_edges > 0`` (cueq's ``env_weighted`` cannot run on empty tensors).
"""

from __future__ import annotations

import pytest
import torch

from simpoly.vivace import constant, keys
from simpoly.vivace.data import prepare_pair_edge_data_lammps_host
from simpoly.vivace.models import VivaceBergamot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-head e0s dict, matching VivaceBergamot.allowed_heads (mirrors
# tests/test_vivace.py).
_BERGAMOT_HEADS = ["cp2k", "orca", "xtb", "omol"]
_E0S = {h: torch.zeros(constant.MAX_ATOMIC_NUMBER + 1, 1) for h in _BERGAMOT_HEADS}

# Use r_max > equiv_r_max so the edge_mask = (edge_length < equiv_r_max) branch
# is actually exercised. When r_max == equiv_r_max the model takes a separate
# all-True mask shortcut (layers.py:251-255) and never hits the guarded branch.
_R_MAX = 5.0
_EQUIV_R_MAX = 2.0


def _make_model(dtype: torch.dtype = torch.float64) -> torch.nn.Module:
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = VivaceBergamot(
            l_max=1,
            parity="o3_full",
            n_invariant_pre_layers=1,
            n_layers=2,
            n_equivariant_features=3,
            n_invariant_features=2,
            n_attn_heads=1,
            eng_mlp_kwargs=dict(hidden_dims=[3]),
            out_scale=0.5,
            use_cuequivariance=torch.cuda.is_available(),
            r_max=_R_MAX,
            equiv_r_max=_EQUIV_R_MAX,
            e0s=_E0S,
        )
    finally:
        torch.set_default_dtype(old)
    return model.to(DEVICE).eval()


def _make_stretched_batch(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """Synthetic LAMMPS-mode batch with all edges OUTSIDE equiv_r_max.

    nlocal=3 atoms, npairs=2 edges. Edge norms = (3.0, 3.5), both inside the
    outer cutoff (r_max=5.0) but both *outside* the equivariant cutoff
    (equiv_r_max=2.0). Therefore ``edge_mask = edge_length < equiv_r_max`` is
    all-False and ``n_equiv_edges == 0``.
    """
    Z = torch.tensor([6, 1, 6], dtype=torch.int64, device=DEVICE) + 1
    edge_vec_raw = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.5, 0.0]], dtype=dtype, device=DEVICE)
    sender = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    receiver = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)
    prep = prepare_pair_edge_data_lammps_host(
        edge_vectors=edge_vec_raw,
        sender=sender,
        receiver=receiver,
    )
    return {
        keys.ATOMIC_NUMBERS: Z,
        keys.NLOCAL: torch.as_tensor(3, dtype=torch.int64, device=DEVICE),
        keys.NTOTAL: torch.as_tensor(3, dtype=torch.int64, device=DEVICE),
        **prep,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuequivariance requires CUDA")
def test_empty_equiv_mask_eager_forward_returns_finite():
    """Pin the contract: stretched-system eager forward must produce finite output.

    With the current shipped guards (`if n_equiv_edges == 0:` branches
    ``layers.py:278/332`` and the early-returns in ``equivariant.py:312/530``),
    this test passes; the empty-branch path supplies zero placeholders. If a
    future change deletes the guards without an OUTER fallback, this test will
    crash inside ``cueq.ltp_cuequivariance`` with
    ``RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1]``.
    """
    model = _make_model()
    batch = _make_stretched_batch()

    # Sanity check: edge_mask is genuinely all-False here.
    edge_lengths = batch[keys.EDGE_LENGTH]
    assert (edge_lengths >= _EQUIV_R_MAX).all(), (
        f"test setup invalid: some edge_length < equiv_r_max={_EQUIV_R_MAX} "
        f"(lengths={edge_lengths.tolist()})"
    )
    assert edge_lengths.shape[0] > 0, "test setup invalid: no raw edges"

    out = model(batch, mode=keys.LAMMPS_MODE)
    forces = out[keys.PAIR_FORCES]
    energy = out[keys.TOTAL_ENERGY]

    # Shape contract: PAIR_FORCES has one entry per *raw* edge (before masking).
    assert forces.shape == (
        edge_lengths.shape[0],
        3,
    ), f"unexpected force shape {tuple(forces.shape)} for {edge_lengths.shape[0]} raw edges"
    assert (
        torch.isfinite(forces).all().item()
    ), f"eager forward produced non-finite forces on stretched (empty-mask) input: {forces}"
    assert (
        torch.isfinite(energy).all().item()
    ), f"eager forward produced non-finite energy on stretched input: {energy}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuequivariance requires CUDA")
def test_empty_equiv_mask_cueq_env_weighted_crashes_on_empty():
    """Document the load-bearing constraint: cueq.env_weighted does NOT support
    empty-edge inputs.

    This is *why* the post- export-friendly path must assume
    ``n_equiv_edges > 0`` and rely on the mliap.py OUTER fallback to dispatch
    stretched inputs to eager. If a future cueq release removes this restriction
    (e.g. fixes the ``x.view(x.size(0), -1)`` ambiguity
    ``ltp_cuequivariance.py:306``), this test will start passing. At which
    point the OUTER fallback can be deleted and the inner branch dropped
    entirely.
    """
    model = _make_model()
    init_header = model.init_features

    # Probe cueq's env_weighted with empty-edge tensors directly.
    n_equivariant_features = init_header.n_equivariant_features
    sph_dim = init_header.sph_dim
    empty_sh = torch.empty((0, sph_dim), device=DEVICE, dtype=torch.float64)
    # The weighted-channels op has its own per-irrep weight layout; just hand it
    # an empty tensor of the right leading dim.
    empty_w = torch.empty((0, n_equivariant_features * sph_dim), device=DEVICE, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"cannot reshape tensor of 0 elements"):
        init_header.env_weighted(empty_sh, empty_w)
