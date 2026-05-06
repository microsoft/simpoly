import dataclasses

import torch

from simpoly.vivace import keys


@dataclasses.dataclass
class PBCRadiusGraphData:
    edge_index: torch.Tensor  # [2, num_edges]
    cell_offsets: torch.Tensor  # [num_edges, 3]
    n_edges_per_graph: torch.Tensor  # [num_structures, ]
    edge_length: torch.Tensor | None  # [num_edges, ]


# cdist ver.
@torch.no_grad()
def get_radius_graph(
    pos: torch.Tensor,  # [n_atoms, 3]
    n_nodes_per_graph: torch.Tensor,  # [n_graphs]
    pbc: torch.Tensor,  # [n_graphs, 3]
    cell: torch.Tensor,  # [n_graphs, 3, 3]
    cutoff_radius: float,
    return_dist: bool = False,
    order_by_distance: bool = True,
) -> PBCRadiusGraphData:

    device = pos.device
    dtype = pos.dtype
    batch_size = len(n_nodes_per_graph)

    assert pbc.dim() == 2 and pbc.shape[1] == 3, "pbc tensor has the wrong shape"
    assert torch.all(pbc[0] == pbc), "PBCs are not equal across batch dimension"
    pbc_ = pbc[0].detach().cpu().numpy().tolist()

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc_[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(cutoff_radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc_[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(cutoff_radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc_[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(cutoff_radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max().item(), rep_a2.max().item(), rep_a3.max().item()]

    # Tensor of unit cells
    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=dtype) for rep in max_rep]
    # [n_cells=8*m1*m2*m3, 3], covering all possible combinations of cells in 3 dimensions
    # potential memory hog here
    cell_offsets = torch.cartesian_prod(*cells_per_dim)
    n_cells = len(cell_offsets)
    # [n_graphs, n_cells, 3]
    unit_cell_batch = cell_offsets.view(1, n_cells, 3).expand(batch_size, -1, -1).contiguous()

    # Compute the x, y, z positional offsets for each cell in each image
    # [n_graphs, n_cells, 3]
    pbc_offsets = torch.bmm(unit_cell_batch, cell)
    # repeat into: [n_atoms, n_cells, 3]
    pbc_offsets_per_atom = pbc_offsets.repeat_interleave(n_nodes_per_graph, dim=0)

    # view: [n_atoms, 1, 3]
    pos_orig_wrapped, shift_to_unwrap = wrap_positions(pos, cell, n_nodes_per_graph, pbc_)

    pos_orig_wrapped = pos_orig_wrapped.view(
        -1, 1, 3
    )  # .expand(-1, n_cells, -1) <-- don't. 10x slower speed awaits
    # broadcast into: [n_atoms, n_cells, 3]
    pos_pbc_shift = pos_orig_wrapped + pbc_offsets_per_atom

    # Compute the distance between atoms.
    @torch.no_grad()
    def dist_thresh(
        A: torch.Tensor, B: torch.Tensor, cutoff: float, _return_dist: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        # note that when data is in float32 format, the cdist function is not stable
        # see https://github.com/msr-ai4science/feynman/issues/12166
        # and https://github.com/pytorch/pytorch/issues/57690
        # therefore, we need to run cdist at float64
        A_prime = A.to(torch.float64)
        B_prime = B.to(torch.float64)
        D = torch.cdist(A_prime, B_prime)
        D = D.to(A.dtype)

        idx = torch.nonzero(torch.logical_and(D < cutoff, D > 0.01), as_tuple=False)
        if not _return_dist:
            return idx
        else:
            values = D[idx[:, 0], idx[:, 1]]
            return idx, values

    @torch.no_grad()
    def blockwise_dist_thresh(
        A: torch.Tensor, B: torch.Tensor, cutoff: float, block_size: int, _return_dist: bool = False
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Iterate over blocks of A and B to compute the pairwise distances between them.
        """

        n, m = A.shape[0], B.shape[0]
        n_blocks = (n + block_size - 1) // block_size
        m_blocks = (m + block_size - 1) // block_size

        ret_idx, ret_val = [], []

        for i in range(n_blocks):
            for j in range(m_blocks):
                A_block = A[i * block_size : (i + 1) * block_size]
                B_block = B[j * block_size : (j + 1) * block_size]

                if not _return_dist:
                    idx = dist_thresh(A_block, B_block, cutoff, _return_dist=False)
                    # idx: [n_edges, 2]
                    idx += torch.tensor([i * block_size, j * block_size], device=device).view(1, 2)
                    ret_idx.append(idx)
                else:
                    idx, val = dist_thresh(A_block, B_block, cutoff, _return_dist=True)
                    # idx: [n_edges, 2]
                    idx += torch.tensor([i * block_size, j * block_size], device=device).view(1, 2)
                    ret_idx.append(idx)
                    ret_val.append(val)
        if not _return_dist:
            return ret_idx
        else:
            return ret_idx, ret_val

    @torch.no_grad()
    def compute_dist_one_graph(
        i: int, j: int, _return_dist: bool = False
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        # [Vi, 3]
        A = pos_orig_wrapped[i:j].reshape(-1, 3).contiguous()
        # [Vi * n_cells, 3]
        B = pos_pbc_shift[i:j].reshape(-1, 3).contiguous()

        # ix, iy, dist: [num_edges]
        # note: ix & iy are 0-based.
        return blockwise_dist_thresh(
            A, B, cutoff_radius, 65536, _return_dist
        )  # 65536 items: 32GB maximum memory consumption per block

    # index offset between images
    graph_end = torch.cumsum(n_nodes_per_graph, dim=0)  # [n_graphs]
    graph_begin = graph_end - (n_nodes_per_graph)

    # note: not cutting with max neighbor now;
    compute_dist = [
        compute_dist_one_graph(
            i, j, _return_dist=return_dist or order_by_distance
        )  # pyright: ignore
        for (i, j) in zip(graph_begin, graph_end)
    ]

    # [[tensor(..), ... total E0], [tensor(..), total E1], ...], len = n_graphs
    # [sender, receiver]: x is in the origin cell, y is origin + ghost nodes
    idx_lst: list[list[torch.Tensor]]
    if return_dist or order_by_distance:
        dist_lst: list[list[torch.Tensor]]
        idx_lst, dist_lst = map(list, zip(*compute_dist))
        # flatten dist
        dist = torch.concat(sum(dist_lst, start=[]))
        order_dist_idx = torch.argsort(dist) if order_by_distance else None
    else:
        idx_lst = compute_dist  # type: ignore[assignment]
        dist = None
        order_dist_idx = None

    # compute E0, 11, ...
    def _compute_nr_edges(edges: list[torch.Tensor]) -> int:
        # edges: [fragments of the edge list]
        return sum(map(len, edges))

    n_neighbors_image = torch.tensor(list(map(_compute_nr_edges, idx_lst)), device=device)
    # flatten index to get 0-based indices:
    # [ [ix0, iy0], [ix1, iy1], ... ]: [sum(Ei), 2]
    index0 = torch.concat(sum(idx_lst, start=[]))
    # iy is in range (0.. sum(Vi * n_cell)) but we need (0..sum(Vi))
    ix = index0[:, 0]
    iy = torch.div(index0[:, 1], n_cells, rounding_mode="floor")
    # get displacement offsets: [sum(Vi)] -> [sum(Ei)] -> [1, sum(Ei)]
    graph_offset = torch.repeat_interleave(graph_begin, n_neighbors_image).view(1, -1)
    edge_index = torch.stack([ix, iy]) + graph_offset
    # pos: [n_atoms, n_cells, 3], and reshaped to [Vi * n_cells, 3] therefore,
    # the indices are arranged like:
    # | [G0] a0c0 a0c1 ... a1c0 a1c1 ... | [G1] a0c0 a0c1 ... a1c0 a1c1 ... |
    # hence, we can easily obtain cell index with iy % n_cell
    cell_offsets_index = index0[:, 1] % n_cells
    cell_offsets = cell_offsets[cell_offsets_index]
    cell_offsets += (
        shift_to_unwrap[edge_index[keys.RECEIVER_INDEX]]
        - shift_to_unwrap[edge_index[keys.SENDER_INDEX]]
    )

    if order_by_distance:
        edge_index = edge_index[:, order_dist_idx]
        cell_offsets = cell_offsets[order_dist_idx]
        if dist is not None:
            dist = dist[order_dist_idx]

    return PBCRadiusGraphData(
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        n_edges_per_graph=n_neighbors_image,
        edge_length=dist if return_dist else None,
    )


def compute_edge_data(
    data: dict[str, torch.Tensor],
    compute_forces: bool = True,
    compute_virial: bool = False,
    normalize: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute distances and displacements for a set of positions considering PBCs."""

    pos = data[keys.POSITIONS]
    edge_index = data[keys.EDGE_INDEX]

    single_graph = False
    # Note: for now are storing missing keys in the data
    if keys.BATCH not in data:
        assert keys.BATCH_PTR not in data
        data[keys.BATCH] = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        data[keys.BATCH_PTR] = torch.tensor([0, pos.shape[0]], dtype=torch.long, device=pos.device)
        single_graph = True
    else:
        if data[keys.BATCH].max() == 0:
            single_graph = True

    batch = data[keys.BATCH]
    n_graphs = data[keys.BATCH_PTR].numel() - 1

    has_cell = keys.CELL in data
    if has_cell:
        cell = data[keys.CELL]
    else:
        cell = torch.empty((0, 3, 3))  # dummy tensor

    if compute_forces:
        pos.requires_grad_()

    cell_displacement = torch.zeros(
        (n_graphs, 3, 3),
        dtype=pos.dtype,
        device=pos.device,
    )

    if compute_virial:
        cell_displacement.requires_grad_()

        # from https://github.com/mir-group/nequip/blob/c56f48fcc9b4018a84e1ed28f762fadd5bc763f1/nequip/nn/_grad_output.py#L263C24-L263C24
        symmetric_displacement = 0.5 * (cell_displacement + cell_displacement.transpose(-1, -2))

        # Positions
        expanded_sd = torch.index_select(symmetric_displacement, 0, batch)
        pos = pos + torch.bmm(pos.unsqueeze(-2), expanded_sd).squeeze(-2)

        # Cell
        if has_cell:
            cell = cell + torch.bmm(cell, symmetric_displacement)

    # Displacement vector pointing from sender -> receiver
    sender, receiver = edge_index[keys.SENDER_INDEX], edge_index[keys.RECEIVER_INDEX]
    vectors = torch.index_select(pos, 0, receiver) - torch.index_select(pos, 0, sender)

    # Offsets (of senders)
    if has_cell:
        edge_cell_shift = data[keys.EDGE_CELL_SHIFT]  # [n_edges, 3]
        if single_graph:
            offsets = torch.einsum("ni,ij->nj", edge_cell_shift, cell.squeeze(0))
        else:
            batch_sender = torch.index_select(batch, 0, sender)
            cell_batch = torch.index_select(cell, 0, batch_sender)
            offsets = torch.einsum("ni,nij->nj", edge_cell_shift, cell_batch)
        vectors = vectors - offsets

    # Compute distances
    eps = 1e-9
    distances = torch.linalg.norm(vectors, dim=-1)  # pylint: disable=not-callable
    if normalize:
        vectors = vectors / (distances.unsqueeze(-1) + eps)

    return {
        keys.EDGE_LENGTH: distances,  # [n_edges, ]
        keys.EDGE_VECTORS: vectors,  # [n_edges, 3]
        keys.CELL_DISPLACEMENT: cell_displacement,  # [n_graphs, 3, 3]
    }


def compute_pair_edge_data_lammps(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # PERF / DYNAMO NOTE: when ``data`` already contains the sorted
    # ``EDGE_LENGTH`` (i.e. ``MLFFModelMLIAP._lammps_data_to_mlff_batch`` has
    # already run the sort + leaf-construction in the *host* uncompiled path),
    # this function is a no-op pass-through. This is the path that
    # ``torch.compile`` actually traces, so it sees a graph with a fresh
    # ``requires_grad=True`` leaf for ``EDGE_VECTORS`` and no ``requires_grad_()``
    # in-place mutation (which dynamo cannot trace; cf. MACE PR #1170).
    if keys.EDGE_LENGTH in data and keys.REVERSE_DIST_SORTING_IDX in data:
        return {
            keys.EDGE_LENGTH: data[keys.EDGE_LENGTH],
            keys.EDGE_VECTORS: data[keys.EDGE_VECTORS],
            keys.MAX_SENDER_IDX: data[keys.NTOTAL],
            keys.MAX_RECEIVER_IDX: data[keys.NLOCAL],
            keys.REVERSE_DIST_SORTING_IDX: data[keys.REVERSE_DIST_SORTING_IDX],
            keys.SENDER: data[keys.SENDER],
            keys.RECEIVER: data[keys.RECEIVER],
        }

    # Slow / fallback / non-MLIAP path (e.g. unit tests calling this function
    # directly on raw input). Equivalent to the legacy implementation.
    vectors = data[keys.EDGE_VECTORS]
    with torch.no_grad():
        distances = torch.linalg.norm(vectors, dim=-1)
        order_dist_idx = torch.argsort(distances)
        reverse_order = torch.argsort(order_dist_idx)
        sorted_vectors = vectors[order_dist_idx]
        sender = data[keys.SENDER][order_dist_idx]
        receiver = data[keys.RECEIVER][order_dist_idx]

    # Construct a fresh leaf with ``requires_grad=True`` (cf. note above:
    # this branch is NOT torch.compile-traced; it's only hit in eager calls).
    new_vectors = sorted_vectors.detach().clone()
    new_vectors.requires_grad_(True)
    distances = torch.linalg.norm(new_vectors, dim=-1)

    return {
        keys.EDGE_LENGTH: distances,
        keys.EDGE_VECTORS: new_vectors,
        keys.MAX_SENDER_IDX: data[keys.NTOTAL],
        keys.MAX_RECEIVER_IDX: data[keys.NLOCAL],
        keys.REVERSE_DIST_SORTING_IDX: reverse_order,
        keys.SENDER: sender,
        keys.RECEIVER: receiver,
    }


def prepare_pair_edge_data_lammps_host(
    edge_vectors: torch.Tensor,
    sender: torch.Tensor,
    receiver: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Host-side (uncompiled) sort + leaf-construction for the LAMMPS path.

    Called by ``MLFFModelMLIAP._lammps_data_to_mlff_batch`` BEFORE the
    compiled forward is invoked. Doing the requires_grad mutation here keeps
    ``torch.compile``'s dynamo away from the unsupported in-place
    ``Tensor.requires_grad_()`` op.
    """
    with torch.no_grad():
        distances = torch.linalg.norm(edge_vectors, dim=-1)
        order_dist_idx = torch.argsort(distances)
        reverse_order = torch.argsort(order_dist_idx)
        sorted_vectors = edge_vectors[order_dist_idx]
        sorted_sender = sender[order_dist_idx]
        sorted_receiver = receiver[order_dist_idx]

    # A leaf tensor with grad-tracking. Constructing in eager (uncompiled) host
    # code, so the in-place ``.requires_grad_()`` is fine here.
    new_vectors = sorted_vectors.detach().clone()
    new_vectors.requires_grad_(True)
    new_distances = torch.linalg.norm(new_vectors, dim=-1)

    return {
        keys.EDGE_VECTORS: new_vectors,
        keys.EDGE_LENGTH: new_distances,
        keys.SENDER: sorted_sender,
        keys.RECEIVER: sorted_receiver,
        keys.REVERSE_DIST_SORTING_IDX: reverse_order,
    }


def compute_pair_edge_data_batch(
    data: dict[str, torch.Tensor],
    compute_grads: bool,
) -> dict[str, torch.Tensor]:
    """Compute distances and displacements for a set of positions considering PBCs."""

    pos = data[keys.POSITIONS]
    n_atoms = torch.as_tensor(pos.shape[0])

    # Displacement vector pointing from sender -> receiver
    edge_index = data[keys.EDGE_INDEX]
    senders, receivers = edge_index[keys.SENDER_INDEX], edge_index[keys.RECEIVER_INDEX]

    # WARNING: we put the sender and receiver indices into the data dict
    data[keys.SENDER] = senders
    data[keys.RECEIVER] = receivers

    # Just to make sure nothing is being recorded
    with torch.no_grad():
        # Vectors = Receiver - Sender
        vectors = torch.index_select(pos, 0, receivers) - torch.index_select(pos, 0, senders)

        # Offsets (of senders)
        if keys.CELL in data:
            batch_sender = torch.index_select(data[keys.BATCH], 0, senders)
            cell_batch = torch.index_select(data[keys.CELL], 0, batch_sender)
            offsets = torch.einsum("ni,nij->nj", data[keys.EDGE_CELL_SHIFT], cell_batch)
            vectors = vectors - offsets

    # Enable gradients for vectors if needed
    if compute_grads:
        vectors.requires_grad_()

    # Compute distances
    distances = torch.linalg.norm(vectors, dim=-1)

    edge_data = {
        keys.EDGE_LENGTH: distances,  # [n_edges, ]
        keys.EDGE_VECTORS: vectors,  # [n_edges, 3]
        keys.MAX_SENDER_IDX: n_atoms,
        keys.MAX_RECEIVER_IDX: n_atoms,
    }

    return edge_data


def wrap_positions(
    pos: torch.Tensor,  # [n_atoms, 3]
    cell: torch.Tensor,  # [n_graphs, 3]
    n_nodes_per_graph: torch.Tensor,  # [n_graphs]
    pbc: list[bool],  # [3]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy from ase.geometry.geometry.wrap"""

    if not any(pbc):
        return pos, torch.zeros_like(pos)

    # by default not changing positions
    shift_T = torch.zeros_like(pos).T

    cell = cell.repeat_interleave(n_nodes_per_graph, dim=0)
    cell_inv = torch.linalg.inv(cell)

    fractional = torch.bmm(pos.unsqueeze(-2), cell_inv)
    fractional = fractional.squeeze(-2)
    fractional_T = fractional.T

    for i, periodic in enumerate(pbc):
        if periodic:
            shift_T[i] = torch.floor(fractional_T[i])
            fractional_T[i] = fractional_T[i] - shift_T[i]

    pos_wrap = torch.bmm(fractional_T.T.unsqueeze(-2), cell)

    return pos_wrap.squeeze(-2), shift_T.T
