"""
convert ase format to our data format. I recycled some old codes I wrote for NequIP. -- Lixin
"""

import copy
import logging
import warnings

import ase
import ase.calculators.singlepoint as ase_sp_calculators
import ase.stress
import numpy as np
import torch

from simpoly.vivace import keys

from .datapoint import MLFFDatapoint

LOG = logging.getLogger(__name__)

ase_to_mlff_key_map: dict[str, str] = {
    "forces": keys.FORCES,
    "energy": keys.TOTAL_ENERGY,
    "energies": keys.PER_ATOM_ENERGY,
    "stress": keys.STRESS,
    "virial": keys.VIRIAL,
    "positions": keys.POSITIONS,
}

mlff_to_ase_key_map = {v: k for k, v in ase_to_mlff_key_map.items()}


def ase_atoms_to_datapoint(
    atoms: ase.Atoms,
    dtype: torch.dtype = torch.float64,
    add_calculator_content: bool = True,
) -> MLFFDatapoint:
    content_dict = {
        keys.ATOMIC_NUMBERS: torch.tensor(atoms.numbers, dtype=torch.int),  # [n_atoms,]
        keys.POSITIONS: torch.tensor(atoms.positions, dtype=dtype),  # [n_atoms, 3]
    }

    # PBC [1, 3] and cell [1, 3, 3]
    content_dict[keys.CELL] = torch.tensor(atoms.get_cell().array, dtype=dtype).unsqueeze(0)
    content_dict[keys.PBC] = torch.tensor(atoms.pbc, dtype=torch.bool).unsqueeze(0)

    # Get info from atoms.arrays; lowest priority. copy first
    add_fields = {
        ase_to_mlff_key_map.get(k.lower(), k): v
        for k, v in atoms.arrays.items()
        if k not in ["numbers", "positions"]
    }

    # Get info from atoms.info; second lowest priority.
    add_fields.update({ase_to_mlff_key_map.get(k.lower(), k): v for k, v in atoms.info.items()})

    if (atoms.calc is not None) and add_calculator_content:
        if isinstance(atoms.calc, ase_sp_calculators.SinglePointCalculator):
            add_fields.update(
                {
                    ase_to_mlff_key_map.get(k.lower, k): copy.deepcopy(v)
                    for k, v in atoms.calc.results.items()
                }
            )
        else:
            raise NotImplementedError(f"Calculator {atoms.calc} not supported")

    # Check if stress and virials are in the add_fields dictionary
    has_stress = keys.STRESS in add_fields
    has_virial = keys.VIRIAL in add_fields

    # This should not happen because ase operates with stress
    if has_virial and not has_stress:
        add_fields[keys.STRESS] = -add_fields[keys.VIRIAL] / atoms.get_volume()
        has_stress = True

    if has_stress:
        # check consistency
        volume = atoms.get_volume()
        if has_virial:
            assert np.allclose(
                add_fields[keys.STRESS], -add_fields[keys.VIRIAL] / volume
            ), "stress and virial are not consistent"

        stress = add_fields[keys.STRESS]

        # reshape stress
        if stress.shape == (1, 3, 3):
            # it's already 3x3, do nothing else
            pass
        elif stress.shape == (3, 3):
            stress = stress.reshape([1, 3, 3])
        elif stress.shape == (6,):
            # it's Voigt order
            stress = ase.stress.voigt_6_to_full_3x3_stress(stress).reshape([1, 3, 3])
        else:
            raise RuntimeError(f"bad shape for {keys.STRESS}: {stress.shape}")

        # final recomputation of virial
        if not has_virial:
            add_fields[keys.VIRIAL] = -stress * volume
        del add_fields[keys.STRESS]

    for k, v in add_fields.items():
        if isinstance(v, torch.Tensor):
            content_dict[k] = v
        elif isinstance(v, np.ndarray):
            if isinstance(v.reshape([-1])[0], np.floating):
                content_dict[k] = torch.tensor(v, dtype=dtype)
            else:
                content_dict[k] = torch.tensor(v)
        elif isinstance(v, int):
            content_dict[k] = torch.tensor([v], dtype=torch.int).reshape([1])
        elif isinstance(v, float):
            content_dict[k] = torch.tensor([v], dtype=dtype).reshape([1])
        elif isinstance(v, list):
            content_dict[k] = torch.tensor(v, dtype=dtype)
        else:
            warnings.warn(f"unrecognized type {type(v)} for key '{k}'")

    return MLFFDatapoint(**content_dict)


def datapoint_to_ase_atoms(datapoint: MLFFDatapoint) -> ase.Atoms:
    atoms = ase.Atoms(
        numbers=datapoint[keys.ATOMIC_NUMBERS],
        positions=datapoint[keys.POSITIONS],
        pbc=datapoint[keys.PBC].squeeze(0) if keys.PBC in datapoint else None,
        cell=datapoint[keys.CELL].squeeze(0) if keys.PBC in datapoint else None,
    )

    # Store energy, forces, and stress
    results = {}
    if keys.TOTAL_ENERGY in datapoint:
        results[mlff_to_ase_key_map[keys.TOTAL_ENERGY]] = (
            datapoint[keys.TOTAL_ENERGY].squeeze(0).numpy()
        )

    if keys.FORCES in datapoint:
        results[mlff_to_ase_key_map[keys.FORCES]] = datapoint[keys.FORCES].numpy()

    # ASE uses stress, not virial
    if keys.VIRIAL in datapoint:
        virial = datapoint[keys.VIRIAL].squeeze(0).numpy()

        # Store stress in the voigt format
        results[mlff_to_ase_key_map[keys.STRESS]] = ase.stress.full_3x3_to_voigt_6_stress(
            -virial / atoms.get_volume()
        )

    calculator = ase_sp_calculators.SinglePointCalculator(atoms, **results)
    atoms.calc = calculator

    return atoms
