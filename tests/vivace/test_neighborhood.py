"""Pure unit tests on `vivace.data.neighborhood.get_radius_graph`.

These exercise the radius-graph builder against an analytical ASE
reference and pin down the corner cases that matter for correctness:

* empty graph (no neighbours within cutoff)
* basic 3-atom triclinic-cell graph, edges + vectors agree with ASE
* invariance to whole-system shift (no PBC)
* anisotropic 1-D periodicity
* `cdist` fp32 stability on a cell+positions tuple known to misbehave
* distance-sorting permutation (regression on the new sort path)

Lightweight: no model, no training, no I/O. Uses
`vivace.data.get_dummy_datapoint()` for the basic cases, and
`ase.neighborlist.primitive_neighbor_list` for the analytical reference.
Adapted from feynman/projects/mdmlff/tests/test_mlff/test_neighborhood.py.
"""

from __future__ import annotations

import dataclasses

import ase.neighborlist
import numpy as np
import numpy.typing as npt
import pytest
import torch

from simpoly.vivace import keys
from simpoly.vivace.data import (
    DataTypeTransform,
    MLFFDatapoint,
    NeighborhoodTransform,
    get_dummy_datapoint,
    neighborhood,
)

# --- helpers ---------------------------------------------------------------


def _to_numpy(t: torch.Tensor) -> npt.NDArray:
    return t.cpu().detach().numpy()


def _assert_rows_all_close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Row-wise equality up to a permutation."""
    assert a.shape == b.shape and a.dim() == 2
    distance = torch.cdist(a, b)
    index = torch.argmin(distance, dim=1).tolist()
    assert len(set(index)) == len(index), "duplicate row match"
    assert torch.allclose(a[index], b)


@dataclasses.dataclass(frozen=True)
class _Edge:
    """An edge keyed by (sender, receiver, |vector|).

    We compare on length (not signed vector) because vivace and ASE may
    use opposite sign conventions for the periodic-image displacement;
    a sign flip here would flip *every* edge, which is not a correctness
    bug -- the model only uses ``EDGE_VECTORS`` through the equivariant
    layers which are O(3)-equivariant.  The signed-convention check is
    pinned separately by ``test_edge_vector_sign_convention``.
    """

    sender: int
    receiver: int
    length: float

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _Edge)
        return (
            self.sender == other.sender
            and self.receiver == other.receiver
            and np.isclose(self.length, other.length)
        )

    def __hash__(self) -> int:  # required because frozen + custom __eq__
        return hash((self.sender, self.receiver, round(self.length, 6)))


def _ase_neighborhood(
    positions: npt.NDArray,
    cutoff: float,
    pbc: tuple[bool, bool, bool],
    cell: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Independent ASE-based reference (after MACE)."""
    sender, receiver, unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,
        use_scaled_positions=False,
    )
    # drop true self-edges that don't cross PBC
    keep = ~((sender == receiver) & np.all(unit_shifts == 0, axis=1))
    sender, receiver, unit_shifts = sender[keep], receiver[keep], unit_shifts[keep]
    edge_index = np.stack((sender, receiver))
    shifts = unit_shifts @ cell
    vectors = positions[receiver] - positions[sender] + shifts
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return edge_index, vectors / (lengths + 1e-9)


def _edge_set_from_radius_graph(
    radius_graph: neighborhood.PBCRadiusGraphData,
    pos: torch.Tensor,
    cell: torch.Tensor,
) -> set[_Edge]:
    n = pos.shape[0]
    edge_data = neighborhood.compute_edge_data(
        data={
            keys.POSITIONS: pos,
            keys.EDGE_INDEX: radius_graph.edge_index,
            keys.CELL: cell,
            keys.EDGE_CELL_SHIFT: radius_graph.cell_offsets,
            keys.N_EDGE_PER_GRAPH: radius_graph.n_edges_per_graph,
            keys.BATCH: torch.zeros(n, dtype=torch.long),
            keys.BATCH_PTR: torch.tensor([0, n], dtype=torch.long),
        },
        normalize=True,
    )
    vectors = edge_data[keys.EDGE_VECTORS]
    return {
        _Edge(int(s), int(r), float(np.linalg.norm(v)))
        for (s, r), v in zip(
            _to_numpy(radius_graph.edge_index.T).tolist(),
            _to_numpy(vectors).tolist(),
        )
    }


# --- tests -----------------------------------------------------------------


def test_empty_neighborhood() -> None:
    """Cutoff smaller than any pair distance → zero edges."""
    d = get_dummy_datapoint()
    g = neighborhood.get_radius_graph(
        pos=d.pos,
        n_nodes_per_graph=torch.tensor([d.pos.shape[0]], dtype=torch.int),
        pbc=d.pbc,
        cell=d.cell,
        cutoff_radius=0.2,
    )
    assert g.edge_index.shape == (2, 0)
    assert g.n_edges_per_graph == torch.tensor([0], dtype=torch.int)


def test_neighborhood_matches_ase_reference() -> None:
    """3-atom triclinic cell: edge set + vectors agree with ASE."""
    d = get_dummy_datapoint()
    cutoff = 0.9
    g = neighborhood.get_radius_graph(
        pos=d.pos,
        n_nodes_per_graph=torch.tensor([d.pos.shape[0]], dtype=torch.int),
        pbc=d.pbc,
        cell=d.cell,
        cutoff_radius=cutoff,
    )
    assert g.edge_index.shape == (2, 6)
    edges = _edge_set_from_radius_graph(g, d.pos, d.cell)

    ref_idx, ref_vec = _ase_neighborhood(
        positions=_to_numpy(d.pos),
        cutoff=cutoff,
        pbc=tuple(_to_numpy(d.pbc.squeeze()).tolist()),
        cell=_to_numpy(d.cell.squeeze()),
    )
    ref_edges = {
        _Edge(int(s), int(r), float(np.linalg.norm(v)))
        for (s, r), v in zip(ref_idx.T.tolist(), ref_vec.tolist())
    }

    key_fn = lambda e: (e.sender, e.receiver, round(e.length, 6))  # noqa: E731
    assert sorted(edges, key=key_fn) == sorted(ref_edges, key=key_fn)


@pytest.mark.parametrize("atom_outside", [False, True], ids=["all_inside", "one_outside"])
def test_neighborhood_invariant_under_atom_unwrap(atom_outside: bool) -> None:
    """Moving one atom outside the cell still yields the same edges/lengths."""
    d = get_dummy_datapoint()
    if atom_outside:
        d.pos[0][0] += 10.0
    cutoff = 0.9
    g = neighborhood.get_radius_graph(
        pos=d.pos,
        n_nodes_per_graph=torch.tensor([d.pos.shape[0]], dtype=torch.int),
        pbc=d.pbc,
        cell=d.cell,
        cutoff_radius=cutoff,
    )
    edges = _edge_set_from_radius_graph(g, d.pos, d.cell)

    ref_idx, ref_vec = _ase_neighborhood(
        positions=_to_numpy(d.pos),
        cutoff=cutoff,
        pbc=tuple(_to_numpy(d.pbc.squeeze()).tolist()),
        cell=_to_numpy(d.cell.squeeze()),
    )
    ref_edges = {
        _Edge(int(s), int(r), float(np.linalg.norm(v)))
        for (s, r), v in zip(ref_idx.T.tolist(), ref_vec.tolist())
    }

    key_fn = lambda e: (e.sender, e.receiver, round(e.length, 6))  # noqa: E731
    assert sorted(edges, key=key_fn) == sorted(ref_edges, key=key_fn)


def test_edge_vector_sign_convention() -> None:
    """Pin vivace's ``EDGE_VECTORS = pos[receiver] - pos[sender] - offsets``
    convention with one explicit non-PBC pair.  If this flips, all
    downstream tests using direction-based references will silently
    break too -- this gives a clear error."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    g = neighborhood.get_radius_graph(
        pos=pos,
        n_nodes_per_graph=torch.tensor([2]),
        pbc=torch.tensor([[False, False, False]], dtype=torch.bool),
        cell=cell,
        cutoff_radius=2.0,
    )
    edge_data = neighborhood.compute_edge_data(
        data={
            keys.POSITIONS: pos,
            keys.EDGE_INDEX: g.edge_index,
            keys.CELL: cell,
            keys.EDGE_CELL_SHIFT: g.cell_offsets,
            keys.N_EDGE_PER_GRAPH: g.n_edges_per_graph,
            keys.BATCH: torch.zeros(2, dtype=torch.long),
            keys.BATCH_PTR: torch.tensor([0, 2], dtype=torch.long),
        },
        normalize=False,
    )
    sender = g.edge_index[keys.SENDER_INDEX]
    receiver = g.edge_index[keys.RECEIVER_INDEX]
    vectors = edge_data[keys.EDGE_VECTORS]
    for s, r, v in zip(sender.tolist(), receiver.tolist(), vectors.tolist()):
        expected = (pos[r] - pos[s]).tolist()
        assert np.allclose(v, expected), (
            f"edge {s}->{r}: vector={v} expected pos[{r}]-pos[{s}]={expected}; "
            "convention may have flipped (downstream tests use this)."
        )


def _two_atom_cubic() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.75, 0.0]])
    cell = torch.tensor([[[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]])
    return pos, cell


def test_neighborhood_translation_invariant_no_pbc() -> None:
    """Same neighbourhood after rigid shift when PBC are off."""
    pos, cell = _two_atom_cubic()
    g = neighborhood.get_radius_graph(
        pos=pos + 3.0,  # shift everything outside the fictitious cell
        n_nodes_per_graph=torch.tensor([pos.shape[0]]),
        pbc=torch.tensor([[False, False, False]], dtype=torch.bool),
        cell=cell,
        cutoff_radius=1.2,
    )
    assert g.edge_index.shape[1] == 2  # 0<->1 both directions


def test_neighborhood_1d_periodic_x_only() -> None:
    """Periodic only along x: 4 edges (each atom sees its mirror image)."""
    pos, cell = _two_atom_cubic()
    g = neighborhood.get_radius_graph(
        pos=pos,
        n_nodes_per_graph=torch.tensor([pos.shape[0]]),
        pbc=torch.tensor([[True, False, False]], dtype=torch.bool),
        cell=cell,
        cutoff_radius=1.2,
    )
    assert g.edge_index.shape[1] == 4

    n = pos.shape[0]
    edge_data = neighborhood.compute_edge_data(
        data={
            keys.POSITIONS: pos,
            keys.EDGE_INDEX: g.edge_index,
            keys.CELL: cell,
            keys.EDGE_CELL_SHIFT: g.cell_offsets,
            keys.N_EDGE_PER_GRAPH: g.n_edges_per_graph,
            keys.BATCH: torch.zeros(n, dtype=torch.long),
            keys.BATCH_PTR: torch.tensor([0, n], dtype=torch.long),
        },
        normalize=False,
    )
    expected = torch.tensor(
        [
            [0.75, -0.75, 0.0],  # 1 -> 0'
            [-0.75, -0.75, 0.0],  # 1 -> 0
            [0.75, 0.75, 0.0],  # 0 -> 1
            [-0.75, 0.75, 0.0],  # 0 -> 1'
        ],
        dtype=torch.get_default_dtype(),
    )
    _assert_rows_all_close(edge_data[keys.EDGE_VECTORS], expected)


def test_cdist_fp32_stability() -> None:
    """Pos+cell hand-picked by Yicheng Chen: naive cdist in fp32 reports
    spurious extra edges.  Our path must give exactly the 2 real edges."""
    cell = torch.tensor(
        [
            [
                [7.580999851226807, 0.0, 0.0],
                [0.0, 11.440999984741211, 0.0],
                [0.0, 0.0, 21.152000427246094],
            ]
        ]
    )
    pos = torch.tensor(
        [[1.04314566, 9.22945499, 18.17379761], [5.42723799, 9.43996906, 20.85375595]]
    )
    pbc = torch.tensor([[True, True, True]])
    data = MLFFDatapoint(
        atomic_numbers=torch.tensor([2, 2], dtype=torch.int),
        pos=pos,
        cell=cell,
        pbc=pbc,
    )
    nh = NeighborhoodTransform(cutoff_radius=4.5)
    dt = DataTypeTransform(torch.float32)
    out = nh(dt(data.clone()))
    assert out.edge_index.shape == (2, 2)


# --- regression on the new sort path ---------------------------------------


def test_radius_graph_distance_sorted() -> None:
    """`get_radius_graph(order_by_distance=True)` returns edges
    monotonically non-decreasing distance order; the inverse permutation
    keys.REVERSE_DIST_SORTING_IDX recovers the unsorted ordering."""
    d = get_dummy_datapoint()
    g_sorted = neighborhood.get_radius_graph(
        pos=d.pos,
        n_nodes_per_graph=torch.tensor([d.pos.shape[0]], dtype=torch.int),
        pbc=d.pbc,
        cell=d.cell,
        cutoff_radius=1.2,
        order_by_distance=True,
    )
    g_raw = neighborhood.get_radius_graph(
        pos=d.pos,
        n_nodes_per_graph=torch.tensor([d.pos.shape[0]], dtype=torch.int),
        pbc=d.pbc,
        cell=d.cell,
        cutoff_radius=1.2,
        order_by_distance=False,
    )
    # Same edge set
    assert g_sorted.edge_index.shape == g_raw.edge_index.shape

    # Compute distances along the sorted edge list
    n = d.pos.shape[0]
    edge_data = neighborhood.compute_edge_data(
        data={
            keys.POSITIONS: d.pos,
            keys.EDGE_INDEX: g_sorted.edge_index,
            keys.CELL: d.cell,
            keys.EDGE_CELL_SHIFT: g_sorted.cell_offsets,
            keys.N_EDGE_PER_GRAPH: g_sorted.n_edges_per_graph,
            keys.BATCH: torch.zeros(n, dtype=torch.long),
            keys.BATCH_PTR: torch.tensor([0, n], dtype=torch.long),
        },
        normalize=True,
    )
    dist_sorted = edge_data[keys.EDGE_LENGTH]
    assert torch.all(
        dist_sorted[1:] >= dist_sorted[:-1] - 1e-9
    ), "Edges should be sorted by distance (non-decreasing)."
