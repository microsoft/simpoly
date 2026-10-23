import pytest
import torch
import torch.nn as nn

from simpoly.vivace import constant, keys
from simpoly.vivace.calculator import MLFFCalculator
from simpoly.vivace.data import (
    ComposedTransform,
    DataTypeTransform,
    NeighborhoodTransform,
    build_tracer_batch,
)
from simpoly.vivace.deploy import load_model, save_model
from simpoly.vivace.models import VivaceBergamot
from simpoly.vivace.models.base import MLFFModel, MLFFMultiHeadedModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-head e0s dict, matching VivaceBergamot.allowed_heads.
_BERGAMOT_HEADS = ["cp2k", "orca", "xtb", "omol"]
_E0S = {h: torch.zeros(constant.MAX_ATOMIC_NUMBER + 1, 1) for h in _BERGAMOT_HEADS}

VIVACE_CONFIG = dict(
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
    r_max=5.0,
    e0s=_E0S,
)


def _make_model():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    model = VivaceBergamot(**VIVACE_CONFIG)
    torch.set_default_dtype(old)
    return model.to(DEVICE)


def _make_batch():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    batch = build_tracer_batch(cutoff_radius=5.0, n_graphs=2)
    transform = ComposedTransform([NeighborhoodTransform(5.0), DataTypeTransform(torch.float64)])
    result = transform(batch)
    torch.set_default_dtype(old)
    d = result.to_dict()
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def test_model_creation():
    model = _make_model()
    assert isinstance(model, nn.Module)
    assert isinstance(model, (MLFFModel, MLFFMultiHeadedModel))
    assert len(list(model.parameters())) > 0


def test_forward_pass():
    model = _make_model()
    batch_dict = _make_batch()
    output = model(batch_dict, compute_forces=True, compute_virial=True)

    assert keys.TOTAL_ENERGY in output
    assert keys.PER_ATOM_ENERGY in output
    assert keys.FORCES in output
    assert keys.VIRIAL in output
    assert output[keys.TOTAL_ENERGY] is not None
    assert output[keys.FORCES] is not None


def test_roundtrip_state_dict(tmp_path):
    model1 = _make_model()
    batch_dict = _make_batch()
    out1 = model1(batch_dict, compute_forces=True, compute_virial=True)

    state_path = tmp_path / "state.pt"
    torch.save(model1.state_dict(), state_path)

    model2 = _make_model()
    model2.load_state_dict(torch.load(state_path, weights_only=True))
    out2 = model2(_make_batch(), compute_forces=True, compute_virial=True)

    assert torch.allclose(out1[keys.TOTAL_ENERGY], out2[keys.TOTAL_ENERGY])
    assert torch.allclose(out1[keys.FORCES], out2[keys.FORCES])
    assert torch.allclose(out1[keys.VIRIAL], out2[keys.VIRIAL])


@pytest.mark.gpu
def test_ase_calculator(tmp_path):
    from ase import Atoms

    model = _make_model()
    model_path = tmp_path / "calc_model.pt"
    save_model(model, model_path)

    calculator = MLFFCalculator(model_path=str(model_path))
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=True,
    )
    atoms.calc = calculator

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (3, 3)
