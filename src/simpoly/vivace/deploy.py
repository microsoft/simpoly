"""Save/load inference-ready Vivace models for the ASE pipeline.

The release artifact is a plain pickled :class:`torch.nn.Module`. Loading is
done with :func:`torch.load` (see :class:`vivace.calculator.MLFFCalculator`).
The companion artifact for LAMMPS is produced by
:func:`vivace.mliap.save_mliap_model`.

Inference performance is provided by ``torch.compile``.
"""

import copy
import functools
import logging
import pathlib
from typing import Any

import torch

from simpoly.vivace import constant, keys
from simpoly.vivace.models.base import MLFFModel

LOG = logging.getLogger(__name__)

dtype_to_str = {
    torch.float32: "float32",
    torch.float64: "float64",
}

dtype_from_str = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _strip_cueq_disable_type_conv(module: torch.nn.Module) -> torch.nn.Module:
    """Remove cuequivariance's ``disable_type_conv`` per-instance tensor attrs.

    ``cuequivariance_torch.primitives.tensor_product.disable_type_conv`` patches
    individual tensor instances with ``t.to = functools.partial(to_notypeconv, t)``
    and ``t.__original_to = t.to``. This is fine while the live cueq module is
    in memory, but pickling a parent model (e.g. via ``torch.save`` after a
    ``copy.deepcopy``) can drop ``__original_to`` while keeping the partial,
    so any later ``t.to(device)`` crashes with
    ``AttributeError: 'Tensor' object has no attribute '__original_to'``.

    Removing both attributes restores the class-level ``Tensor.to`` for those
    buffers. Safe for inference / deployment because the FX graph itself never
    calls ``.to()`` during forward; the partial only mattered as a guard
    against external ``model.to(dtype=...)`` mutations, which we don't do
    after the model is prepared for deployment.

    Tested against cuequivariance==0.2.0 (originally) and 0.8.x (current
    Dockerfile pin); the ``disable_type_conv`` pattern is unchanged across
    that range.
    """
    for _, buf in module.named_buffers():
        d = buf.__dict__
        if isinstance(d.get("to"), functools.partial):
            d.pop("to", None)
        d.pop("__original_to", None)
    return module


def _prepare_model_for_deploy(
    model: torch.nn.Module,
    compile_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Deep-copy, strip cueq partials, freeze, eval, and optionally compile."""
    prepared = copy.deepcopy(model)
    _strip_cueq_disable_type_conv(prepared)
    for param in prepared.parameters():
        param.requires_grad = False
    prepared.eval()
    if compile_kwargs is not None:
        compiled = torch.compile(prepared, **compile_kwargs)
        assert isinstance(compiled, torch.nn.Module)
        return compiled
    return prepared


def build_deployment_metadata(
    is_edge_centered: bool,
    r_max: float,
    lammps_cutoff: float,
    dtype: torch.dtype,
    allow_tf32: bool | None = None,
) -> dict[str, Any]:
    """Build the deployment metadata dictionary stored alongside an ASE model."""
    dtype_str = dtype_to_str[dtype]

    return {
        keys.IS_EDGE_CENTERED: bool(is_edge_centered),
        keys.R_MAX: float(r_max),
        keys.LAMMPS_CUTOFF: float(lammps_cutoff),
        keys.N_SPECIES: len(constant.CHEMICAL_SYMBOLS),
        keys.TYPE_NAMES: list(constant.CHEMICAL_SYMBOLS),
        keys.DEFAULT_DTYPE: dtype_str,
        keys.MODEL_DTYPE: dtype_str,
        keys.TF32: bool(allow_tf32) if allow_tf32 is not None else False,
    }


class MLFFModelASE:
    """Inference-ready container for the ASE pipeline.

    Mirrors the role of :class:`vivace.mliap.MLFFModelMLIAP`: a plain
    pickle-friendly Python object that holds the live ``nn.Module`` plus the
    deployment metadata. Loaded by :class:`vivace.calculator.MLFFCalculator`.
    """

    def __init__(
        self,
        model: MLFFModel,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        instance_meta = model.get_instance_metadata()
        cls_meta = model.get_cls_metadata()

        self.model = _prepare_model_for_deploy(model, compile_kwargs)
        self.metadata = build_deployment_metadata(
            is_edge_centered=cls_meta.is_edge_centered,
            r_max=instance_meta.r_max,
            lammps_cutoff=instance_meta.lammps_cutoff,
            dtype=instance_meta.dtype,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def save_model(
    model: MLFFModel,
    path: str | pathlib.Path,
    compile_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save a Vivace model in the ASE-side ``.pt`` format."""
    ase_model = MLFFModelASE(model, compile_kwargs=compile_kwargs)
    LOG.info(f"Saving ASE deployment model to '{path}' with metadata: {ase_model.metadata}")
    torch.save(ase_model, str(path))


def load_model(
    path: str | pathlib.Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a Vivace ASE-side ``.pt`` produced by :func:`save_model`.

    Returns the live ``nn.Module`` (already on ``device`` and in ``eval`` mode)
    and the deployment metadata dictionary.
    """
    ase_model = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    if not isinstance(ase_model, MLFFModelASE):
        raise TypeError(
            f"Expected a vivace.deploy.MLFFModelASE pickle at '{path}'; "
            f"got {type(ase_model).__name__}."
        )
    model = ase_model.model.to(device)
    model.eval()

    metadata = dict(ase_model.metadata)
    metadata[keys.DEFAULT_DTYPE] = dtype_from_str[metadata[keys.DEFAULT_DTYPE]]
    return model, metadata
