# SPDX-License-Identifier: MIT

import dataclasses
import typing as ty

import ase.calculators.lammps as ase_lammps
import ase.data
import ase.io.lammpsdata
import numpy as np
from rdkit.Chem import AllChem


def count_real_atoms(smiles: str) -> int:
    """
    Count the number of real atoms in a molecule.

    This function takes a SMILES string and counts all atoms with non-zero
    atomic numbers in the corresponding molecule. This excludes any special
    pseudo-atoms or query features that might have atomic number zero.
    """

    # edge-case
    if smiles == "H*":
        return 1

    mol = AllChem.AddHs(AllChem.MolFromSmiles(smiles, sanitize=True))  # type: ignore
    return sum(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())


@dataclasses.dataclass(frozen=True, slots=True)
class PolymerStats:
    n_tot: int
    n_chains: int
    n_ru_per_chain: int


def compute_polymer_stats(
    n_tot: int | None,
    n_chains: int | None,
    n_ru_per_chain: int | None,
    ru_smiles: str,
    end_group_smiles: tuple[str, str],
) -> PolymerStats:
    assert (n_tot is None) + (n_chains is None) + (
        n_ru_per_chain is None
    ) == 1, "Exactly one of n_tot, n_chains, and n_ru_per_chain must be None."

    n_ru_atoms = count_real_atoms(ru_smiles)
    n_end_atoms = sum(count_real_atoms(smiles) for smiles in end_group_smiles)

    if n_tot is None:
        assert n_chains is not None and n_ru_per_chain is not None
        n_atoms_per_chain = n_ru_atoms * n_ru_per_chain + n_end_atoms
        n_tot = n_chains * n_atoms_per_chain
    elif n_chains is None:
        assert n_ru_per_chain is not None and n_tot is not None
        n_atoms_per_chain = n_ru_atoms * n_ru_per_chain + n_end_atoms
        n_chains = int(np.floor(n_tot / n_atoms_per_chain))
    else:
        assert n_ru_per_chain is None and n_tot is not None and n_chains is not None
        n_atoms_per_chain = n_tot // n_chains
        n_ru_per_chain = (n_atoms_per_chain - n_end_atoms) // n_ru_atoms

    return PolymerStats(n_tot=n_tot, n_chains=n_chains, n_ru_per_chain=n_ru_per_chain)


def get_masses_section(elements: list[str], units: str) -> str:
    lines = []
    for t, s in enumerate(elements):
        mass = ase.data.atomic_masses[ase.data.atomic_numbers[s]]
        mass = ase_lammps.convert(mass, quantity="mass", fromunits="ASE", tounits=units)  # type: ignore
        lines.append(f"mass {t + 1:>6} {mass:23.17g} # {s}")
    return "\n".join(lines)


def get_atomic_number_from_mass(atoms: ase.Atoms) -> list[np.signedinteger[ty.Any]]:
    """Data file in LAMMPS distinguishes atomic type according by their masses.
    The masses read in LAMMPS files are not the same as those in `ase.data`.
    Thus, the atomic number will be determined the mass that is the closest.
    """
    return [np.argmin(np.abs(ase.data.atomic_masses - mass)) for mass in atoms.get_masses()]


def get_specorder_from_atoms(atoms: ase.Atoms) -> list[str]:
    """This function assumes the 'type' field is populated in the ase.Atoms object."""
    zs = atoms.get_atomic_numbers()  # type: ignore
    types = atoms.arrays["type"]

    assert zs.shape == types.shape
    assert zs.ndim == 1

    # It can be that there a type in the LAMMPS data file that is not present in the atoms object.
    # For instance, there could be 3 types, but only the first and the third are used.
    # The default atomic number is 0, i.e., the chemical element "X".
    t_to_z = {t: 0 for t in range(1, max(types) + 1)}

    for t, z in zip(types, zs):
        t_to_z[t] = z

    return [ase.data.chemical_symbols[z] for z in t_to_z.values()]


def rewrite_full_to_metal_data(in_path: str, out_path: str) -> list[str]:
    atoms = ase.io.lammpsdata.read_lammps_data(in_path, atom_style="full", units="real")
    assert isinstance(atoms, ase.Atoms)
    atoms.set_atomic_numbers(get_atomic_number_from_mass(atoms))  # type: ignore
    atom_types = get_specorder_from_atoms(atoms)
    ase.io.lammpsdata.write_lammps_data(
        out_path,
        atoms,
        atom_style="atomic",
        units="metal",
        specorder=atom_types,
    )
    return atom_types
