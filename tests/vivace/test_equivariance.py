"""SE(3) equivariance check on the public ``vivace_v0.1`` checkpoint.

Adapted from feynman/projects/mdmlff/tests/test_mlff/test_models/test_symmetry.py,
but pinned to the deployed JIT instead of an untrained tiny model.  Using the
real model gives much higher signal: a real architectural break in the
equivariant pipeline (e.g. a missing tensor-product output) would shift the
forces by O(1) eV/A.

For a random rotation R and translation t applied to all atom positions
(and to the cell when PBC is on):

* Energy is **invariant**:  E(R x + t) == E(x)
* Forces are **equivariant**:  F(R x + t) == R F(x)
* Stress (3x3) transforms by **conjugation**:  S' == R S R^T

We do not check virial separately because vivace's ASE calculator already
exposes stress, and stress = -virial / V on a fixed-volume rotation.
"""

from __future__ import annotations

import pathlib

import ase
import ase.stress
import numpy as np
import pytest
from ase.io import read

try:  # pragma: no cover
    import cuequivariance_ops_torch  # noqa: F401
except Exception:  # pragma: no cover
    pass


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CHECKPOINT = REPO / "checkpoints" / "vivace_v0.1.pt"
POLYMER_XYZ = HERE / "data" / "test_polymer.lmps"


@pytest.fixture
def polymer_atoms() -> ase.Atoms:
    # Hard-coded type->Z map for tests/vivace/data/test_polymer.lmps:
    # type 1 = C (Z=6), type 2 = H (Z=1).
    return read(str(POLYMER_XYZ), format="lammps-data", Z_of_type={1: 6, 2: 1})


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CHECKPOINT.exists(),
        reason=f"Public vivace_v0.1 checkpoint not present at {CHECKPOINT}.",
    ),
]


def _random_rotation(seed: int) -> np.ndarray:
    """Random 3x3 rotation via QR of a Gaussian matrix (Haar-distributed
    after sign correction)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _evaluate(atoms: ase.Atoms, calc) -> tuple[float, np.ndarray, np.ndarray]:
    atoms.calc = calc
    E = float(atoms.get_potential_energy())
    F = np.asarray(atoms.get_forces(), dtype=np.float64)
    S = ase.stress.voigt_6_to_full_3x3_stress(
        np.asarray(atoms.get_stress(voigt=True), dtype=np.float64)
    )
    return E, F, S


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_se3_equivariance(seed: int, polymer_atoms: ase.Atoms) -> None:
    """E invariant, F transforms as R F, stress as R S R^T under rigid
    rotation + translation of the system."""
    from simpoly.vivace.calculator import MLFFCalculator

    R = _random_rotation(seed)
    t = np.array([0.37, -1.4, 2.1])  # small translation to also test invariance

    base = polymer_atoms
    rot = base.copy()
    rot.set_positions(base.get_positions() @ R.T + t)
    if any(base.pbc):
        rot.set_cell(np.asarray(base.get_cell()) @ R.T)

    calc = MLFFCalculator(model_path=str(CHECKPOINT))
    E0, F0, S0 = _evaluate(base, calc)
    E1, F1, S1 = _evaluate(rot, calc)

    # Empirical noise floor on vivace_v0.1 + 32-atom polymer (3 random
    # rotations measured): energy is bit-exact; forces drift up to
    # ~0.14 eV/A on small-magnitude components with RMS ~0.03 (fp32
    # atomicAdd nondeterminism in cuequivariance kernels); stress
    # drifts up to ~2e-5 eV/A^3.  Bounds set ~2x measured.

    # Energy invariance (in practice bit-exact)
    np.testing.assert_allclose(
        E1,
        E0,
        atol=1e-4,
        rtol=1e-7,
        err_msg=f"Energy not invariant under rotation: dE = {E1 - E0:.3e}",
    )

    # Force equivariance: F(R x) should equal R F(x)
    F0_rot = F0 @ R.T
    f_diff = F1 - F0_rot
    f_max = float(np.abs(f_diff).max())
    f_rms = float(np.sqrt((f_diff**2).mean()))
    assert (
        f_max < 0.25 and f_rms < 0.06
    ), f"Forces not equivariant: max={f_max:.3e}, rms={f_rms:.3e} eV/A"

    # Stress equivariance: S' = R S R^T
    S0_rot = R @ S0 @ R.T
    s_max = float(np.abs(S1 - S0_rot).max())
    assert s_max < 5e-5, f"Stress not equivariant: max abs diff = {s_max:.3e} eV/A^3"


def test_translation_invariance(polymer_atoms: ase.Atoms) -> None:
    """E, F, stress all invariant under pure translation (no rotation).
    Tighter tolerance than the rotation test since no equivariant
    transform is applied to outputs."""
    from simpoly.vivace.calculator import MLFFCalculator

    base = polymer_atoms
    shifted = base.copy()
    shifted.set_positions(base.get_positions() + np.array([2.5, -0.4, 1.7]))

    calc = MLFFCalculator(model_path=str(CHECKPOINT))
    E0, F0, S0 = _evaluate(base, calc)
    E1, F1, S1 = _evaluate(shifted, calc)

    np.testing.assert_allclose(E1, E0, atol=1e-4, rtol=1e-7)
    f_diff = F1 - F0
    f_max = float(np.abs(f_diff).max())
    f_rms = float(np.sqrt((f_diff**2).mean()))
    assert (
        f_max < 0.2 and f_rms < 0.05
    ), f"Force translation drift: max={f_max:.3e}, rms={f_rms:.3e}"
    s_max = float(np.abs(S1 - S0).max())
    assert s_max < 5e-5, f"Stress translation drift: max={s_max:.3e}"
