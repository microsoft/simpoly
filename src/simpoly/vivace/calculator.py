from typing import Any, Final, Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from simpoly.vivace import keys
from simpoly.vivace.data import (
    ComposedTransform,
    DataTypeTransform,
    NeighborhoodTransform,
    ase_atoms_to_datapoint,
)
from simpoly.vivace.deploy import load_model

MODEL_PATH: Final[str] = "model_path"


class MLFFCalculator(Calculator):  # type: ignore[misc]
    # required for ase calculator interface
    implemented_properties = ["energy", "energies", "forces", "stress"]

    # typing for MLFF specific variable
    model: torch.nn.Module
    r_max: float
    dtype: torch.dtype

    def __init__(self, model_path: str, **kwargs: dict[str, Any]) -> None:
        Calculator.__init__(self, **kwargs)  # pyright: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, metadata = load_model(model_path, device=self.device)

        self.dtype = metadata[keys.DEFAULT_DTYPE]
        r_max = metadata[keys.R_MAX]

        self.transform = ComposedTransform(
            [NeighborhoodTransform(cutoff_radius=r_max), DataTypeTransform(dtype=self.dtype)]
        )
        self.model = model

    def atoms_to_dict(self, atoms: Atoms) -> dict[str, torch.Tensor]:
        data_point = ase_atoms_to_datapoint(atoms, add_calculator_content=False)
        data_point = self.transform(data_point)
        for k in data_point.keys():
            if isinstance(data_point[k], torch.Tensor):
                data_point[k] = data_point[k].to(self.device)
        d = data_point.to_dict()
        # Add batch metadata for single-graph input
        n_atoms = len(atoms)
        if keys.BATCH not in d:
            d[keys.BATCH] = torch.zeros(n_atoms, dtype=torch.long, device=self.device)
        if keys.BATCH_PTR not in d:
            d[keys.BATCH_PTR] = torch.tensor([0, n_atoms], dtype=torch.long, device=self.device)
        return d

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list[str]] = None,
        system_changes: list[str] = all_changes,
    ) -> None:

        if properties is None:
            properties = self.implemented_properties

        if len(system_changes) == 0 and len(self.results) > 0:
            if all([p in self.results for p in properties]):
                return

        # make a copy of atoms to self.atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        compute_virial = "stress" in properties
        compute_forces = "forces" in properties

        assert self.atoms is not None
        self.atoms.wrap()
        data = self.atoms_to_dict(self.atoms)
        output = self.model(data, compute_virial=compute_virial, compute_forces=compute_forces)
        self.results["energy"] = output[keys.TOTAL_ENERGY].item()
        self.results["energies"] = output[keys.PER_ATOM_ENERGY].detach().cpu().numpy()

        if compute_forces:
            self.results["forces"] = output[keys.FORCES].detach().cpu().numpy()

        if compute_virial:
            virial = output[keys.VIRIAL].detach().cpu().numpy()
            # if not defining cell in atoms
            cell = self.atoms.get_cell()
            volume = 1.0
            if np.abs(cell.array).sum() > 0:
                volume = self.atoms.get_volume()
            self.results["stress"] = -full_3x3_to_voigt_6_stress(virial.squeeze(0)) / volume
