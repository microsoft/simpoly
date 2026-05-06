import pathlib

import numpy as np
import pytest
from ase.io import read

# 1 eV/Å³ in bar (CODATA elementary charge; 1 Pa = 1e-5 bar).
EV_PER_A3_TO_BAR = 1.602176634e6

# pp_*.lmps atom-type → atomic number, matching the LAMMPS pair_coeff
# ``* * C H`` used in tests/test_lammps_mliap.py (type 1 = C, type 2 = H).
Z_OF_TYPE = {1: 6, 2: 1}


@pytest.mark.gpu
@pytest.mark.parametrize("n_atoms", [274])
def test_ase_calculator_matches_lammps_reference(
    test_data_dir: pathlib.Path,
    test_checkpoint_ase_pt: pathlib.Path,
    n_atoms: int,
) -> None:
    if not test_checkpoint_ase_pt.exists():
        pytest.skip(f"ASE checkpoint not present at {test_checkpoint_ase_pt}.")

    from simpoly.vivace.calculator import MLFFCalculator

    data = test_data_dir / f"pp_{n_atoms}.lmps"
    ref_npz = test_data_dir / f"pp_{n_atoms}_lammps_reference.npz"

    atoms = read(str(data), format="lammps-data", Z_of_type=Z_OF_TYPE)
    assert len(atoms) == n_atoms

    calc = MLFFCalculator(model_path=str(test_checkpoint_ase_pt))
    atoms.calc = calc

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress_voigt = np.asarray(atoms.get_stress(voigt=True), dtype=float)

    # Convert ASE stress (eV/Å³, tensile-positive, [xx,yy,zz,yz,xz,xy])
    # → LAMMPS thermo pressure (bar, compressive-positive, [xx,yy,zz,xy,xz,yz]).
    ase_xx, ase_yy, ase_zz, ase_yz, ase_xz, ase_xy = stress_voigt
    pressure_bar = -EV_PER_A3_TO_BAR * np.array([ase_xx, ase_yy, ase_zz, ase_xy, ase_xz, ase_yz])

    reference = np.load(ref_npz)
    assert n_atoms == int(reference["n_atoms"])

    e_ref = float(reference["energy_eV"])
    f_ref = reference["forces_eV_per_A"]
    p_ref = reference["pressure_bar"]

    # Tolerances mirror tests/test_lammps_mliap.py: the ASE pipeline is
    # numerically equivalent to the LAMMPS pipeline on the same atoms, so
    # the only allowable slack is the LAMMPS reference's own print precision
    # (energy / pressure at thermo precision; forces at dump "%g" ≈ 6
    # significant figures → ~1e-5 absolute on per-component forces of O(1)).
    # Measured on pp_274: |ΔE| 5e-5 eV, max|ΔF| 5e-5 eV/Å, max|ΔP| 0.15 bar.
    assert np.isclose(energy, e_ref, rtol=1e-6), f"energy mismatch: {energy} vs {e_ref}"
    assert np.allclose(
        forces, f_ref, atol=5e-5
    ), f"force mismatch: max|ΔF|={np.abs(forces - f_ref).max():.4e} eV/Å"
    assert np.allclose(
        pressure_bar, p_ref, atol=1.0, rtol=1e-4
    ), f"pressure mismatch: max|ΔP|={np.abs(pressure_bar - p_ref).max():.4e} bar"
