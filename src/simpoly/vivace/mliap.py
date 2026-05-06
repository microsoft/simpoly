"""MLIAP interface for Vivace models in LAMMPS.

Usage:
    pair_style mliap unified <model_path> 0
    pair_coeff * * <atom_types>

The model_path should point to a file created by vivace.mliap.save_mliap_model().

This module assumes a LAMMPS-enabled environment (the ``vivace-lammps:cu13``
docker image), so ``cupy`` and ``lammps.mliap.mliap_unified_abc`` are imported
unconditionally at module load. Users who don't need LAMMPS should simply not
import :mod:`vivace.mliap`.
"""

import logging
import os
import typing as ty

import ase.data
import cupy as cp
import torch
from lammps.mliap.mliap_unified_abc import MLIAPUnified

from simpoly.vivace import constant, keys
from simpoly.vivace.data import prepare_pair_edge_data_lammps_host
from simpoly.vivace.deploy import _prepare_model_for_deploy
from simpoly.vivace.models.base import MLFFModel  # noqa: E402

LOG = logging.getLogger(__name__)

# Set a more allocator-friendly default before any CUDA context is created.
# ``expandable_segments`` reduces fragmentation cost across the LAMMPS step
# loop (where npairs drifts every neighbor-list rebuild). Use ``setdefault``
# so callers / Dockerfile can still override at the process level.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set a more allocator-friendly default before any CUDA context is created.
# ``expandable_segments`` reduces fragmentation cost across the LAMMPS step
# loop (where npairs drifts every neighbor-list rebuild). Use ``setdefault``
# so callers / Dockerfile can still override at the process level.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Default torch.compile kwargs for MLIAP inference: enable Inductor codegen
# with dynamic shapes so a small npairs drift between neighbor-list rebuilds
# doesn't trigger a recompile. Override via
# ``MLFFModelMLIAP(compile_kwargs=...)`` at construction, or by setting
# the env var ``VIVACE_MLIAP_DISABLE_COMPILE=1`` to fall back to eager.
DEFAULT_COMPILE_KWARGS: dict[str, ty.Any] = {
    "mode": "default",
    "dynamic": True,
    "fullgraph": False,
}


def _resolve_compile_kwargs(
    compile_kwargs: dict[str, ty.Any] | None,
) -> dict[str, ty.Any] | None:
    """Resolve compile_kwargs with sentinel and env-var override.

    - ``None`` (default) → ``DEFAULT_COMPILE_KWARGS``
    - ``{}`` (empty dict) → disables compile (sentinel: explicit opt-out)
    - env ``VIVACE_MLIAP_DISABLE_COMPILE=1`` → disables compile
    - env ``VIVACE_MLIAP_COMPILE_MODE=<str>`` overrides the inductor mode
      (e.g. ``"max-autotune-no-cudagraphs"``). Cudagraph modes are unsafe
      here (see PR-#1 commit message), so the env knob explicitly does NOT
      accept ``reduce-overhead``/``max-autotune`` (which both bake
      cudagraphs).
    - anything else → returned as-is
    """
    if os.environ.get("VIVACE_MLIAP_DISABLE_COMPILE", "0") == "1":
        return None
    if compile_kwargs is None:
        compile_kwargs = dict(DEFAULT_COMPILE_KWARGS)
    elif not compile_kwargs:
        return None
    else:
        compile_kwargs = dict(compile_kwargs)

    env_mode = os.environ.get("VIVACE_MLIAP_COMPILE_MODE", "").strip()
    if env_mode:
        if env_mode in ("reduce-overhead", "max-autotune"):
            LOG.warning(
                "VIVACE_MLIAP_COMPILE_MODE=%s requested but is unsafe with "
                "the dynamic npairs path (cudagraph re-capture); ignoring.",
                env_mode,
            )
        else:
            compile_kwargs["mode"] = env_mode
    return compile_kwargs


def save_mliap_model(
    model: MLFFModel,
    path: str,
    compile_kwargs: dict[str, ty.Any] | None = None,
) -> None:
    """Save a Vivace model in MLIAP format for LAMMPS."""
    mliap_model = MLFFModelMLIAP(model, compile_kwargs=compile_kwargs)
    torch.save(mliap_model, path)


class MLFFModelMLIAP(MLIAPUnified):  # type: ignore[misc]
    """Vivace integration for LAMMPS using the MLIAP unified interface.

    See https://github.com/lammps/lammps/blob/develop/python/lammps/mliap/mliap_unified_abc.py
    """

    def __init__(self, model: MLFFModel, compile_kwargs: dict[str, ty.Any] | None = None) -> None:
        super().__init__()

        # Resolve compile_kwargs once, then defer the actual ``torch.compile``
        # call to ``_initialize_device`` so it runs *after* ``.to(device)``
        # (cudagraphs / Inductor needs the model already on the target device).
        # We still strip cueq partials + freeze + eval here.
        self._compile_kwargs = _resolve_compile_kwargs(compile_kwargs)
        self.model = _prepare_model_for_deploy(model, compile_kwargs=None)
        self._compiled = False
        self.dtype = model.get_instance_metadata().dtype
        self.device = torch.device("cpu")
        self.initialized = False
        self.element_types = [
            ase.data.chemical_symbols[s] for s in range(1, constant.MAX_ATOMIC_NUMBER + 1)
        ]
        self.ndescriptors = 1
        self.nparams = 1
        # LAMMPS doubles the cutoff given for MLIAP
        self.rcutfac = 0.5 * model.get_instance_metadata().r_max

    def _maybe_compile(self) -> None:
        """Apply ``torch.compile`` lazily, after the model is on its target device.

        Robust to old pickles that pre-date this attribute (uses ``getattr``).
        """
        if getattr(self, "_compiled", False):
            return
        # Old pickles won't have ``_compile_kwargs``; fall back to default.
        ckwargs = getattr(self, "_compile_kwargs", None)
        if ckwargs is None and not hasattr(self, "_compile_kwargs"):
            ckwargs = _resolve_compile_kwargs(None)
        if ckwargs:
            # Enable safe dynamo capture knobs so .item() / boolean-mask
            # ops (radial_clamping, layers InitHeader, vivace_bergamot) fold back into
            # the compiled graph instead of triggering graph breaks. These are
            # numerically equivalent (do NOT enable trace_autograd_ops, which
            # produces wrong forces with cuequivariance kernels).
            import torch._dynamo as _dynamo

            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.capture_dynamic_output_shape_ops = True
            # Optional Inductor codegen knobs (env-gated A/B). Off by default
            # because the prior PR-#1 measurement only validated the default
            # mode. Enable via VIVACE_MLIAP_INDUCTOR_TUNING=1.
            if os.environ.get("VIVACE_MLIAP_INDUCTOR_TUNING", "0") == "1":
                import torch._inductor.config as _ind

                _ind.epilogue_fusion = True
                _ind.coordinate_descent_tuning = True
                LOG.info(
                    "VIVACE_MLIAP_INDUCTOR_TUNING=1 → "
                    "epilogue_fusion=True, coordinate_descent_tuning=True"
                )
            LOG.info(f"Compiling MLIAP model with torch.compile(**{ckwargs})")
            self.model = torch.compile(self.model, **ckwargs)
        self._compiled = True

    def _initialize_device(self, data) -> None:  # type: ignore[no-untyped-def]
        if isinstance(data.elems, cp.ndarray):
            device = torch.device(f"cuda:{data.elems.device.id}")
        else:
            device = torch.device("cpu")
        self.device = device
        self.model = self.model.to(device)
        # Enable cuDNN benchmark for autotuned conv kernels.
        # Note: TF32 matmul (allow_tf32 / matmul_precision="high") was *tried*
        # but breaks the accuracy gate (max|ΔF|≈5.9e-3 vs tolerance 2e-5);
        # the test contract is locked, so we keep fp32 matmul and only enable
        # the (numerically-neutral) cuDNN autotuner here.
        # An opt-in escape hatch ``VIVACE_MLIAP_TF32=1`` is provided for users
        # whose accuracy contract is looser than the default test gate; off
        # by default.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if os.environ.get("VIVACE_MLIAP_TF32", "0") == "1":
                torch.set_float32_matmul_precision("high")
                LOG.warning(
                    "VIVACE_MLIAP_TF32=1 → set_float32_matmul_precision('high'); "
                    "this exceeds the default 2e-5 force tolerance. Caller "
                    "must validate accuracy."
                )
        # Compile AFTER moving to device so cudagraphs / Inductor see the
        # final placement (the ``mode="reduce-overhead"`` path captures into
        # a CUDA graph keyed on the device the parameters live on).
        self._maybe_compile()
        LOG.info(f"Vivace MLIAP model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data) -> None:  # type: ignore[no-untyped-def]
        if not self.initialized:
            self._initialize_device(data)
        # NOTE: even with no edges we'd still need the model call for E0s,
        # but if there are no local atoms there is nothing to update.
        if data.nlocal == 0:
            return

        batch = self._lammps_data_to_mlff_batch(data)
        props = self.model(batch, mode=keys.LAMMPS_MODE)

        energy = props[keys.TOTAL_ENERGY].detach()
        pair_forces = props[keys.PAIR_FORCES].detach()
        self._update_lammps_data(data, pair_forces, energy)

        # Note: sync moved from BEFORE `_update_lammps_data` to AFTER it.
        # The captured graph will replay end-to-end without an
        # internal sync; the LAMMPS handoff still requires that the GPU
        # writes (`update_pair_forces_gpu` via cupy zero-copy view of the
        # forces tensor in `_update_lammps_data`) have completed before
        # control returns to the LAMMPS C++ integrator. Placing the sync
        # here preserves that contract while removing the in-graph sync.
        if self.device.type != "cpu":
            torch.cuda.synchronize()

    def _lammps_data_to_mlff_batch(self, data) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
        # See https://github.com/lammps/lammps/blob/7ca493917a9e9c4f3da0625802ebaa63022602d5/src/KOKKOS/mliap_unified_couple_kokkos.pyx#L150
        # - data.rij : distance vector from i (sender) to j (receiver) [npairs, 3]
        # - data.pair_i/j : i/j index of each ij pair [npairs]
        # - data.nlocal : number of local atoms
        # - data.elems : index of chemical symbols in self.element_types
        # i in [0, nlocal), j in [0, ntotal) (incl. ghosts) -> use i as receiver,
        # j as sender, and flip the sign of the LAMMPS edge vector accordingly.
        #
        # Do the distance-sort + leaf-construction (with ``requires_grad=True``)
        # HERE in the uncompiled host code rather than inside the compiled forward.
        # ``torch.compile`` / dynamo cannot trace ``Tensor.requires_grad_()``
        # in-place mutation; doing it in the host pipeline lets the model forward
        # see a pre-prepared graph with no graph break.
        edge_vectors = -torch.as_tensor(data.rij, dtype=self.dtype, device=self.device)
        sender = torch.as_tensor(data.pair_j, dtype=torch.int64, device=self.device)
        receiver = torch.as_tensor(data.pair_i, dtype=torch.int64, device=self.device)

        prepared = prepare_pair_edge_data_lammps_host(
            edge_vectors=edge_vectors,
            sender=sender,
            receiver=receiver,
        )

        return {
            keys.ATOMIC_NUMBERS: torch.as_tensor(data.elems, dtype=torch.int64, device=self.device)
            + 1,
            keys.NLOCAL: torch.as_tensor(data.nlocal, dtype=torch.int64, device=self.device),
            keys.NTOTAL: torch.as_tensor(data.ntotal, dtype=torch.int64, device=self.device),
            **prepared,
        }

    def _update_lammps_data(self, data, pair_forces: torch.Tensor, energy: torch.Tensor) -> None:  # type: ignore[no-untyped-def]
        # LAMMPS expects double precision
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
            energy = energy.double()

        if self.device.type == "cpu":
            data.update_pair_forces(pair_forces.numpy())
            data.energy = energy.numpy()
        else:
            data.update_pair_forces_gpu(cp.asarray(pair_forces))
            data.energy = cp.asarray(energy)

    def compute_descriptors(self, data) -> None:  # type: ignore[no-untyped-def]
        return

    def compute_gradients(self, data) -> None:  # type: ignore[no-untyped-def]
        return
