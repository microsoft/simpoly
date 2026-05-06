"""Copy from NequIP https://github.com/mir-group/nequip/blob/3fd2213ac3b35f9254ca6e934431d3a81fd64701/nequip/data/keys.py


Revision:
- almost everything is deleted
- _keys.py content is folded into this single module.
"""

from typing import Final, Literal

SPLIT_TYPES = Literal["train", "val", "test"]
TRAIN: Final[SPLIT_TYPES] = "train"
VAL: Final[SPLIT_TYPES] = "val"
TEST: Final[SPLIT_TYPES] = "test"

_SPECIAL_IRREPS = [None]

RECEIVER_INDEX: Final[int] = 0
SENDER_INDEX: Final[int] = 1

# the order of sender and receiver is hard coded in lines that are marked as "assuming the first atom is the receiver atom" in vivace.data.neighbor_list.get_radius_graph
assert (SENDER_INDEX == 1) and (
    RECEIVER_INDEX == 0
), "SENDER_INDEX must be greater than RECEIVER_INDEX due to the hard coded order in get_radius_graph"


# == Define allowed keys as constants ==
# The positions of the atoms in the system
POSITIONS: Final[str] = "pos"
# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX: Final[str] = "edge_index"
RECEIVER: Final[str] = "receiver"
SENDER: Final[str] = "sender"
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT: Final[str] = "cell_offsets"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
CELL: Final[str] = "cell"
# [n_batch, 3] bool tensor
PBC: Final[str] = "pbc"
# [n_atom, 1] long tensor
ATOMIC_NUMBERS: Final[str] = "atomic_numbers"
# [n_atom, 1] long tensor
ATOM_TYPE: Final[str] = "atom_types"
NLOCAL: Final[str] = "nlocal"
NTOTAL: Final[str] = "ntotal"
MAX_RECEIVER_IDX: Final[str] = "max_receiver_index"
MAX_SENDER_IDX: Final[str] = "max_sender_index"
REAL_INDEX: Final[str] = "lammps_real_index"
IS_REAL: Final[str] = "lammps_is_real_atom"
THREEBODY_EDGE_INDEX: Final[str] = "threebody_edge_indices"

# the charges name need to be consistent with ai4s-qc
# https://github.com/msr-ai4science/feynman/projects/ai4s-qc/src/ai4s_qc/orca/output.py#L112
XTB_MULLIKEN_CHARGES = "mulliken_charges"
MULLIKEN_ATOMIC_CHARGES: Final[str] = "mulliken_atomic_charges"
LOEWDIN_ATOMIC_CHARGES: Final[str] = "loewdin_atomic_charges"

N_EDGES_PER_GRAPH: Final[str] = "n_edges_per_graph"

# these keys need to be consistent with the argument names in vivace.data.datapoint.MLFFDataPoint
BASIC_STRUCTURES: Final[list[str]] = [
    POSITIONS,
    EDGE_INDEX,
    EDGE_CELL_SHIFT,
    CELL,
    PBC,
    ATOM_TYPE,
    ATOMIC_NUMBERS,
]

# auxiliary variables for GNN
CELL_DISPLACEMENT = "cell_displacement"
ATTENTION_BIAS_FIELD: Final[str] = "attention_bias"
ACTIVE_EDGES: Final[str] = "active_edges"


NODE_FEATURES: Final[str] = "node_features"
NODE_ATTRS: Final[str] = "node_attrs"

N_EDGE_PER_GRAPH = "n_edges_per_graph"
EDGE_VECTORS: Final[str] = "vectors"
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH: Final[str] = "distances"
REVERSE_DIST_SORTING_IDX: Final[str] = "reverse_dist_sorting_idx"
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS: Final[str] = "edge_attrs"
# [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING: Final[str] = "edge_embedding"
EDGE_EMBEDDING_EQUIV_RMAX: Final[str] = "edge_embedding_equi_rmax"
EDGE_FEATURES: Final[str] = "edge_features"
# [n_edge, 1] invariant of the radial cutoff envelope for each edge, allows reuse of cutoff envelopes
EDGE_LENGTH_ENVELOPE: Final[str] = "edge_length_envelope"
# edge mask for the short equivalent edges
EDGE_MASK: Final[str] = "edge_mask"
EDGE_LENGTH_ENVELOPE_EQUI: Final[str] = "edge_length_envelope_equi"
# edge energy as in Allegro
EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_LATENT_FIELD: Final[str] = "edge_latent_feature"
EDGE_EQUIVARIANT: Final[str] = "edge_equivariant_feature"
EDGE_INVARIANT: Final[str] = "edge_invariant_feature"
EDGE_TYPE_EMBED: Final[str] = "edge_type_embedding"
NODE_TYPE_EMBED: Final[str] = "node_type_embedding"
NODE_EQUIVARIANT: Final[str] = "node_equivariant_feature"
NODE_INVARIANT: Final[str] = "node_invariant_feature"
EDGE_SCALAR_FIELD: Final[str] = "edge_scalar_feature"
# edge attention as in Allegro
EDGE_ATTENTION: Final[str] = "edge_attention"

# typical MLFF output quantities
# they need to be consistent with the C++ Lammps interface
PER_ATOM_ENERGY: Final[str] = "per_atom_energy"
TOTAL_ENERGY: Final[str] = "energy"
TOTAL_CHARGE: Final[str] = "total_charge"
PAIR_FORCES: Final[str] = "pair_forces"
FORCES: Final[str] = "forces"
PARTIAL_FORCES: Final[str] = "partial_forces"
STRESS: Final[str] = "stress"
VIRIAL: Final[str] = "virial"

PER_ATOM_ENERGY_ENSEMBLE: Final[str] = "per_atom_energy_ensemble"
TOTAL_ENERGY_ENSEMBLE: Final[str] = "energy_ensemble"
FORCES_ENSEMBLE: Final[str] = "forces_ensemble"
VIRIAL_ENSEMBLE: Final[str] = "virial_ensemble"

DIPOLE_MOMENT: Final[str] = "dipole_moment"

# total energy divided by number of atoms, for loss calculation
ENERGY_PER_ATOM: Final[str] = "energy_per_atom"

ALL_ENERGYS: Final[list[str]] = [
    EDGE_ENERGY,
    PER_ATOM_ENERGY,
    TOTAL_ENERGY,
    FORCES,
    PARTIAL_FORCES,
    STRESS,
    VIRIAL,
]

NUM_GRAPHS: Final[str] = "num_graphs"
BATCH: Final[str] = "batch"
BATCH_PTR: Final[str] = "ptr"
MULTI_GRAPH_FIELDS: Final[list[str]] = [BATCH, BATCH_PTR, NUM_GRAPHS]

# typical deploy metadata keys

R_MAX: Final[str] = "r_max"
LAMMPS_CUTOFF: Final[str] = "lammps_cutoff"
LAMMPS_TAG: Final[str] = "lammps_tag"
LAMMPS_NUMNEIGH: Final[str] = "lammps_numneigh"
LAMMPS_NEIGHBORS: Final[str] = "lammps_neighbors"
LAMMPS_ILIST: Final[str] = "lammps_ilist"
IS_EDGE_CENTERED: Final[str] = "is_edge_centered"
N_SPECIES: Final[str] = "n_species"
TYPE_NAMES: Final[str] = "type_names"
TF32: Final[str] = "allow_tf32"
DEFAULT_DTYPE: Final[str] = "default_dtype"
MODEL_DTYPE: Final[str] = "model_dtype"

_ALL_METADATA = [
    IS_EDGE_CENTERED,
    LAMMPS_CUTOFF,
    R_MAX,
    N_SPECIES,
    TYPE_NAMES,
    TF32,
    DEFAULT_DTYPE,
    MODEL_DTYPE,
]

_NODE_FIELDS = [
    POSITIONS,
    ATOMIC_NUMBERS,
    ATOM_TYPE,
    FORCES,
    NODE_FEATURES,
    NODE_ATTRS,
    PER_ATOM_ENERGY,
    MULLIKEN_ATOMIC_CHARGES,
    LOEWDIN_ATOMIC_CHARGES,
]
_EDGE_FIELDS = [
    EDGE_CELL_SHIFT,
    EDGE_LENGTH,
    EDGE_ATTRS,
    EDGE_EMBEDDING,
    EDGE_LENGTH_ENVELOPE,
    EDGE_ENERGY,
    EDGE_VECTORS,
    EDGE_CELL_SHIFT,
]
_GRAPH_FIELDS = [
    CELL,
    PBC,
    TOTAL_ENERGY,
    TOTAL_CHARGE,
    DIPOLE_MOMENT,
    N_EDGE_PER_GRAPH,
]
ALLOWED = BASIC_STRUCTURES + _NODE_FIELDS + _EDGE_FIELDS + _GRAPH_FIELDS
# EDGE_INDEX is excluded because it's shape is [2, n_edge]

# Statistics
E0S: Final[str] = "e0s"
AVG_N_NEIGHBORS: Final[str] = "avg_n_neighbors"

BATCH_MODE: Final[str] = "batch"
LAMMPS_MODE: Final[str] = "lammps"
