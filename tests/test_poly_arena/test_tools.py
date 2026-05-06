# SPDX-License-Identifier: MIT

import pathlib

import ase.io.lammpsdata
import numpy as np
import pytest

from simpoly.poly_arena.simulation import tools as sim_tools


@pytest.mark.parametrize(
    "smiles, count",
    [
        ("C", 5),
        ("*C*", 3),  # 2 H atoms are added
        ("CC", 8),
        ("*CC*", 6),  # 2 H atoms are added
        ("O", 3),
        ("H*", 1),
    ],
)
def test_atom_count(smiles: str, count: int) -> None:
    assert sim_tools.count_real_atoms(smiles) == count


def test_polymer_stats() -> None:
    # Case 1
    n_ru_per_chain = 3
    n_chains = 2
    stats = sim_tools.compute_polymer_stats(
        n_tot=None,
        n_chains=n_chains,
        ru_smiles="*C*",
        end_group_smiles=("O*", "H*"),
        n_ru_per_chain=n_ru_per_chain,
    )
    assert stats.n_tot == n_chains * (3 * n_ru_per_chain + 2 + 1)
    assert stats.n_chains == n_chains

    # Case 2
    n_tot = 100
    stats = sim_tools.compute_polymer_stats(
        n_tot=n_tot,
        n_chains=None,
        ru_smiles="*C*",
        end_group_smiles=("O*", "H*"),
        n_ru_per_chain=n_ru_per_chain,
    )
    assert stats.n_tot == n_tot
    assert stats.n_chains == np.floor(n_tot / (3 * n_ru_per_chain + 2 + 1))


def test_masses_section() -> None:
    elements = ["H", "C"]
    section = sim_tools.get_masses_section(elements, units="metal")  # g/mol
    expected = (
        """mass      1      1.0079999997406976 # H\nmass      2      12.010999996910238 # C"""
    )
    assert section == expected


def test_mass_to_atomic_number() -> None:
    path = pathlib.Path(__file__).parent / "assets" / "test_data.lmps"
    atoms = ase.io.lammpsdata.read_lammps_data(path, units="real")
    atomic_numbers = sim_tools.get_atomic_number_from_mass(atoms)
    assert np.allclose(atomic_numbers[:8], np.array([6, 1, 1, 6, 1, 6, 1, 1]))


def test_specorder_from_atoms() -> None:
    path = pathlib.Path(__file__).parent / "assets" / "test_data.lmps"
    atoms = ase.io.lammpsdata.read_lammps_data(path, units="real")
    atoms.set_atomic_numbers(sim_tools.get_atomic_number_from_mass(atoms))
    specorder = sim_tools.get_specorder_from_atoms(atoms)
    assert specorder == ["C", "C", "H"]
