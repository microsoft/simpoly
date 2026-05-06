from typing import Any, Optional, TypeAlias

import torch
import torch_geometric.data

from simpoly.vivace import keys

TensorDict: TypeAlias = dict[str, torch.Tensor]
TensorDictOptional: TypeAlias = dict[str, Optional[torch.Tensor]]


class MLFFDatapoint(torch_geometric.data.Data):  # type: ignore
    r"""A MLFFDatapoint is a Pytorch Geometric Data object describing a MLFF molecular graph with
    atoms in 3D space. The data object can hold graph-level attributes, as well as (pre-computed)
    edge information. In general, :class:`~torch_geometric.data.Data` tries to mimic the behavior
    of a regular Python dictionary. In addition, it provides useful functionality for analyzing
    graph structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`__
    for the accompanying tutorial.

    Args:
        atomic_numbers (LongTensor): Atomic numbers following ase.Atom, (Unknown=0, H=1) with shape
            :obj:`[num_nodes]`. (default: :obj:`None`)
        pos (Tensor): Node position matrix, only set one position value. Unit=Angstrom.
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        pbc (BoolTensor, optional): Periodic Boundary Conditions
            :obj:`[1, 3]`. (default: :obj:`None`)
        cell (Tensor, optional): Cell matrix if pbc = True. Unit=Angstrom.
            :obj:`[1, 3, 3]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Edge indexes (sender, receiver)
            :obj:`[2, num_edges]`. (default: :obj:`None`)
        cell_offsets (IntTensor, optional): Which periodic image does the end of the edge belong to.
            :obj:`[num_edges, 3]`. (default: :obj:`None`)
        energy (Tensor, optional): Graph-level energy label. Unit=eV.
            :obj:`[1]`. (default: :obj:`None`)
        forces (Tensor, optional): Node forces matrix. Unit=eV/Angstrom.
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        virial (Tensor, optional): Graph virial matrix with shape
            :obj:`[1, 3, 3]`. (default: :obj:`None`)
    """

    mandatory_keys = [keys.ATOMIC_NUMBERS, keys.POSITIONS]

    # Add type annotations to reduce number of pyright warnings
    # Technically the following attributes are not mandatory though
    pos: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor

    additional_keys = [
        keys.PBC,
        keys.CELL,
        keys.EDGE_INDEX,
        keys.EDGE_CELL_SHIFT,
        keys.TOTAL_ENERGY,
        keys.FORCES,
        keys.VIRIAL,
    ]
    recognized_keys = mandatory_keys + additional_keys + keys.MULTI_GRAPH_FIELDS

    def __init__(
        self,
        atomic_numbers: torch.Tensor,  # int, [n_atoms]
        pos: torch.Tensor,  # float, [n_atoms, 3]
        pbc: Optional[torch.Tensor] = None,  # bool, [1, 3]
        cell: Optional[torch.Tensor] = None,  # float, [1, 3, 3]
        edge_index: Optional[torch.Tensor] = None,  # long, [2, n_edges]
        cell_offsets: Optional[torch.Tensor] = None,  # int, [n_edges, 3]
        energy: Optional[torch.Tensor] = None,  # float, [1]
        forces: Optional[torch.Tensor] = None,  # float, [n_nodes, 3]
        virial: Optional[torch.Tensor] = None,  # float, [1, 3, 3]
        **kwargs: Any,
    ) -> None:
        super().__init__(edge_index=edge_index, pos=pos, **kwargs)
        # Edge index
        if self.edge_index is not None:
            assert (
                isinstance(self.edge_index, torch.Tensor)
                and self.edge_index.dim() == 2
                and self.edge_index.shape[0] == 2
                and self.edge_index.dtype == torch.long
            )

        # Note: when datapoints are collated empty datapoints are created in which there are no
        # positions or atomic numbers

        # Positions and atomic numbers
        n_atoms: int | None = None
        dtype: torch.dtype | None = None
        if self.pos is not None or atomic_numbers is not None:
            assert (
                isinstance(self.pos, torch.Tensor)
                and self.pos.dim() == 2
                and self.pos.shape[1] == 3
            )
            n_atoms = self.pos.shape[0]
            dtype = self.pos.dtype

            assert (
                isinstance(atomic_numbers, torch.Tensor)
                and atomic_numbers.shape == (n_atoms,)
                and atomic_numbers.dtype == torch.int
            )
            self.atomic_numbers = atomic_numbers

        # Cell and PBC
        if cell is not None or pbc is not None:
            assert (
                isinstance(cell, torch.Tensor) and cell.shape == (1, 3, 3) and cell.dtype == dtype
            )
            self.cell = cell

            assert isinstance(pbc, torch.Tensor) and pbc.shape == (1, 3) and pbc.dtype == torch.bool
            self.pbc = pbc

        # Cell offsets
        if cell_offsets is not None:
            assert self.edge_index is not None
            assert (
                isinstance(cell_offsets, torch.Tensor)
                and cell_offsets.shape == (self.edge_index.shape[1], 3)
                and cell_offsets.dtype == dtype
            )
            self.cell_offsets = cell_offsets

        # Energy
        if energy is not None:
            assert (
                isinstance(energy, torch.Tensor) and energy.shape == (1,) and energy.dtype == dtype
            )
            self.energy = energy

        # Forces
        if forces is not None:
            assert (
                isinstance(forces, torch.Tensor)
                and forces.shape == (n_atoms, 3)
                and forces.dtype == dtype
            )
            self.forces = forces

        # Virial
        assert keys.STRESS not in kwargs, "Use virial instead of stress"
        if virial is not None:
            assert (
                isinstance(virial, torch.Tensor)
                and virial.shape == (1, 3, 3)
                and virial.dtype == dtype
            )
            self.virial = virial


def get_dummy_datapoint() -> MLFFDatapoint:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.7, 0.0],
            [0.0, 0.8, 0.7],
        ],
        dtype=torch.get_default_dtype(),
    )

    cell = torch.tensor(
        [
            [
                [1.0, 0.2, 0.0],  # a
                [0.0, 2.0, 0.0],  # b
                [0.0, 0.0, 1.5],  # c
            ]
        ],
        dtype=torch.get_default_dtype(),
    )

    forces = torch.tensor(
        [
            [1.2, 0.3, -0.5],
            [1.7, 1.8, 1.1],
            [1.9, 1.8, 1.1],
        ],
        dtype=torch.get_default_dtype(),
    )

    virial = torch.tensor(
        [
            [
                [1.0, 0.5, 0.3],
                [0.5, -2.0, 0.0],
                [0.3, 0.0, 0.4],
            ]
        ],
        dtype=torch.get_default_dtype(),
    )

    return MLFFDatapoint(
        atomic_numbers=torch.tensor([1, 6, 6], dtype=torch.int),
        pos=pos,
        pbc=torch.tensor([[True, True, True]], dtype=torch.bool),
        cell=cell,
        energy=torch.tensor([-4.2], dtype=torch.get_default_dtype()),
        forces=forces,
        virial=virial,
    )
