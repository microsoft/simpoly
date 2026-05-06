from typing import List, Optional

import torch

from simpoly.vivace import keys

from . import scatter


@torch.no_grad()
def compute_volume(
    cell: torch.Tensor,  # [n_graphs, 3, 3]
) -> torch.Tensor:
    # the det(cell) is positive if and only if the cell vectors follow the right-hand rule
    # so we can just take the absolute value to avoid negative volume
    volume = torch.det(cell).abs()  # [n_graphs, ]
    if (volume < 1e-8).any():
        raise RuntimeError("Found zero volume")
    return volume


def compute_forces_property(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    # torch.jit is very meticulous with type annotations
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(  # [n_nodes, 3]
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,  # type: ignore[arg-type]
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[0]

    if gradient is None:
        return torch.zeros_like(positions)

    return -1 * gradient


def compute_virial_property(
    energy: torch.Tensor,
    displacement: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(  # [n_graphs, 3, 3]
        outputs=[energy],  # [n_graphs, ]
        inputs=[displacement],  # [n_graphs, 3, 3]
        grad_outputs=grad_outputs,  # type: ignore[arg-type]
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )[0]

    if gradient is None:
        return torch.zeros_like(displacement)

    return -1 * gradient


def compute_forces_and_virial(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    pos_gradient, disp_gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3], [n_graphs, 3, 3]
        grad_outputs=grad_outputs,  # type: ignore[arg-type]
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )

    if pos_gradient is None:
        pos_gradient = torch.zeros_like(positions)

    if disp_gradient is None:
        disp_gradient = torch.zeros_like(displacement)

    return -1 * pos_gradient, -1 * disp_gradient


def compute_force_field_properties(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    compute_forces: bool,
    compute_virial: bool,
    training: bool,
    per_atom_energy: Optional[torch.Tensor] = None,
) -> dict[str, Optional[torch.Tensor]]:
    results: dict[str, Optional[torch.Tensor]] = {keys.TOTAL_ENERGY: energy}
    if compute_forces and compute_virial:
        forces, virial = compute_forces_and_virial(
            energy=energy,
            positions=positions,
            displacement=displacement,
            training=training,
        )  # [n_nodes, 3], [n_graphs, 3, 3]
        results[keys.FORCES] = forces
        results[keys.VIRIAL] = virial
    elif compute_forces:
        forces = compute_forces_property(
            energy=energy,
            positions=positions,
            training=training,
        )  # [n_nodes, 3]
        results[keys.FORCES] = forces
        results[keys.VIRIAL] = None
    elif compute_virial:
        virial = compute_virial_property(
            energy=energy,
            displacement=displacement,
            training=training,
        )
        results[keys.FORCES] = None
        results[keys.VIRIAL] = virial
    else:
        results[keys.FORCES] = None
        results[keys.VIRIAL] = None

    if per_atom_energy is not None:
        results[keys.PER_ATOM_ENERGY] = per_atom_energy

    return results


def get_pair_forces(  # named get_pair_forces to avoid collision with compute_pair_forces kwarg
    energy: torch.Tensor,  # [n_graphs,]
    vectors: torch.Tensor,  # [n_edges, 3]
    training: bool,
) -> torch.Tensor:
    # torch.jit is very meticulous with type annotations
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(  # [n_nodes, 3]
        outputs=[energy],  # [n_graphs, ]
        inputs=[vectors],  # [n_edges, 3]
        grad_outputs=grad_outputs,  # type: ignore[arg-type]
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[0]

    if gradient is None:
        return torch.zeros_like(vectors)

    return -1 * gradient  # [n_edges, 3]


def compute_forces_from_pair_forces(
    f_ij: torch.Tensor,  # [n_edges, 3]
    receiver: torch.Tensor,  # [n_edges,]
    sender: torch.Tensor,  # [n_edges,]
    n_nodes: int,
) -> torch.Tensor:

    f_i = scatter.scatter_sum(  # [n_nodes, 3]
        src=f_ij,
        index=receiver,
        dim=0,
        dim_size=n_nodes,
    )

    f_j = scatter.scatter_sum(  # [n_nodes, 3]
        src=f_ij,
        index=sender,
        dim=0,
        dim_size=n_nodes,
    )

    # v = receiver - sender
    # therefore, the sender gets a -1
    f_tot = f_i - f_j  # [n_nodes, 3]

    return f_tot


def compute_virial_from_pair_forces(
    f_ij: torch.Tensor,  # [n_edges, 3]
    vectors: torch.Tensor,  # v_ij, [n_edges, 3]
    n_graphs: int,
    n_edges_per_graph: torch.Tensor,  # [n_graphs,]
) -> torch.Tensor:
    s_pij = torch.einsum("pi,pj->pij", vectors, f_ij)  # [n_edges, 3, 3]

    r = torch.arange(n_graphs, device=f_ij.device)  # [n_graphs,]
    idx = torch.repeat_interleave(r, repeats=n_edges_per_graph, dim=0)  # [n_edges,]

    s_ij = scatter.scatter_sum(  # [n_graphs, 3, 3]
        src=s_pij,
        index=idx,
        dim=0,
        dim_size=n_graphs,
    )

    # Symmetrize the tensor
    s_sym = 0.5 * (s_ij + s_ij.transpose(-1, -2))  # [n_graphs, 3, 3]

    return s_sym


def compute_pair_properties_batch(
    energy: torch.Tensor,  # [n_graphs,]
    vectors: torch.Tensor,  # [n_edges, 3], where n_edges can be 0
    receiver: torch.Tensor,  # [n_edges,], the receiver atom index for each edge
    sender: torch.Tensor,  # [n_edges,], the sender atom index for each
    n_nodes: int,
    n_graphs: int,
    n_edges_per_graph: torch.Tensor,  # [n_graphs,]
    compute_forces: bool,
    compute_virial: bool,
    training: bool,
) -> dict[str, Optional[torch.Tensor]]:
    results: dict[str, Optional[torch.Tensor]] = {keys.TOTAL_ENERGY: energy}

    if not compute_forces and not compute_virial:
        # Nothing to compute, just return the energy
        return results

    # Needed for forces and virial
    f_ij = get_pair_forces(  # [n_edges, 3]
        energy=energy,
        vectors=vectors,
        training=training,
    )

    if compute_forces:
        results[keys.FORCES] = compute_forces_from_pair_forces(  # [n_nodes, 3]
            f_ij=f_ij,  # [n_edges, 3]
            receiver=receiver,
            sender=sender,
            n_nodes=n_nodes,
        )

    if compute_virial:
        results[keys.VIRIAL] = compute_virial_from_pair_forces(  # [n_graphs,]
            f_ij=f_ij,  # [n_edges, 3]
            vectors=vectors,  # v_ij, [n_edges, 3]
            n_graphs=n_graphs,  # [n_graphs,]
            n_edges_per_graph=n_edges_per_graph,
        )

    return results


def compute_pair_properties_lammps(
    energy: torch.Tensor,  # [n_graphs,]
    vectors: torch.Tensor,  # [n_edges, 3], where n_edges can be 0
    reverse_dist_sorting_idx: torch.Tensor,  # [n_edges,]
) -> dict[str, Optional[torch.Tensor]]:
    forces = get_pair_forces(  # [n_edges, 3]
        energy=energy,
        vectors=vectors,
        training=False,
    )
    return {
        keys.TOTAL_ENERGY: energy,
        keys.PAIR_FORCES: forces[reverse_dist_sorting_idx],
    }


def compute_force_field_properties_ensemble(
    energy_list: List[
        torch.Tensor
    ],  # need to use List as opposed to list for torch.jit compatibility
    positions: torch.Tensor,
    displacement: torch.Tensor,
    compute_forces: bool,
    compute_virial: bool,
    per_atom_energy_list: Optional[
        List[torch.Tensor]
    ] = None,  # need to use List as opposed to list for torch.jit compatibility
) -> dict[str, Optional[torch.Tensor]]:

    results: dict[str, Optional[torch.Tensor]] = {}

    forces_list: List[torch.Tensor] = []
    virial_list: List[torch.Tensor] = []
    for energy in energy_list:
        if compute_forces and compute_virial:
            forces, virial = compute_forces_and_virial(
                energy=energy,
                positions=positions,
                displacement=displacement,
                training=True,  # needs to retain graph even during eval for shallow ensemble
            )  # [n_nodes, 3], [n_graphs, 3, 3]
            forces_list.append(forces)
            virial_list.append(virial)
        elif compute_forces:
            forces = compute_forces_property(
                energy=energy,
                positions=positions,
                training=True,  # needs to retain graph even during eval for shallow ensemble
            )
            forces_list.append(forces)
        elif compute_virial:
            virial = compute_virial_property(
                energy=energy,
                displacement=displacement,
                training=True,  # needs to retain graph even during eval for shallow ensemble
            )
            virial_list.append(virial)

    energy_ensemble = torch.stack(energy_list, dim=0)  # [n_ensembles, n_graphs]
    results[keys.TOTAL_ENERGY_ENSEMBLE] = energy_ensemble
    results[keys.TOTAL_ENERGY] = torch.mean(energy_ensemble, dim=0)  # [n_graphs]

    if per_atom_energy_list is not None:
        per_atom_energy_ensemble = torch.stack(per_atom_energy_list, dim=0).unsqueeze(
            -1
        )  # [n_ensembles, n_nodes, 1]
        results[keys.PER_ATOM_ENERGY_ENSEMBLE] = per_atom_energy_ensemble
        results[keys.PER_ATOM_ENERGY] = torch.mean(per_atom_energy_ensemble, dim=0)  # [n_nodes, 1]

    if compute_forces:
        forces_ensemble = torch.stack(forces_list, dim=0)  # [n_ensembles, n_nodes, 3]
        results[keys.FORCES_ENSEMBLE] = forces_ensemble
        results[keys.FORCES] = torch.mean(forces_ensemble, dim=0)  # [n_nodes, 3]
    else:
        results[keys.FORCES] = None
        results[keys.FORCES_ENSEMBLE] = None

    if compute_virial:
        virial_ensemble = torch.stack(virial_list, dim=0)  # [n_ensembles, n_graphs, 3, 3]
        results[keys.VIRIAL_ENSEMBLE] = virial_ensemble
        results[keys.VIRIAL] = torch.mean(virial_ensemble, dim=0)  # [n_graphs, 3, 3]
    else:
        results[keys.VIRIAL] = None
        results[keys.VIRIAL_ENSEMBLE] = None

    return results
