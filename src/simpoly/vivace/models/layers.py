import math
from typing import Any, Final, Literal, Optional

import cuequivariance_torch
import torch
from e3nn import o3

import simpoly.vivace.modules.ltp_cuequivariance as cue
from simpoly.vivace import constant, keys
from simpoly.vivace.modules import (
    MLP,
    AtomicEmbedding,
    BesselBasis,
    CosineClamp,
    ExpNormalSmearing,
    GaussianBasis,
    MakeWeightedChannels,
    NoBiasMLP,
    PolynomialClamp,
    QKVConcat,
    QKVExp,
    QKVSiLu,
    QKVSoftmax,
    SmoothBesselBasis,
    StepClamp,
)
from simpoly.vivace.modules.radial_basis_set import MixExpNormSmoothBessel
from simpoly.vivace.utils.scatter import scatter_sum

# Note:(AOTI exportability): module-level flag toggled by
# `vivace.mliap` immediately before invoking `torch.export.export` /
# `aoti_compile_and_package`. When True, the data-dependent `if mask.sum()==0`
# / `if n_equiv_edges==0` empty-edge guards in `InitHeader.forward` and the
# equivariant attention layers collapse to their *non-empty* always-take path
# (with `torch._check(...) > 0` annotations so the exporter can resolve the
# previously unbacked SymInts). The shipped `torch.compile` and eager paths
# leave this False and keep the original guards intact, so neither runtime
# regresses. The AOTI dispatcher in `mliap.py` is responsible for routing
# stretched (mask-all-False) inputs to eager so the export-friendly path's
# implicit invariant `n_equiv_edges > 0` actually holds at runtime.
_AOTI_EXPORT_MODE: bool = False


def set_aoti_export_mode(value: bool) -> None:
    """Toggle the export-friendly branch in `InitHeader` / equivariant layers.

    Intended sole caller: `vivace.mliap._maybe_load_aoti` around the
    `torch.export.export` + `aoti_compile_and_package` calls.
    """
    global _AOTI_EXPORT_MODE
    _AOTI_EXPORT_MODE = bool(value)


def is_aoti_export_mode() -> bool:
    return _AOTI_EXPORT_MODE


O3_FULL: Final[str] = "o3_full"
O3_RESTRICTED: Final[str] = "o3_restricted"
SO3: Final[str] = "so3"
PARITY_SETTING_TYPES = Literal["o3_full", "o3_restricted", "so3"]


class InitHeader(torch.nn.Module):
    """
    Initializes the features for the Vivace model.

    This module is responsible for computing the initial node and edge features
    from the atomic numbers and positions. It creates:
    - Invariant node features (from atomic embeddings).
    - Invariant edge features (from radial basis functions and atomic embeddings).
    - Equivariant edge features (or node features if `use_node_equivariant` is True)
      by weighting spherical harmonics with learned weights.
    - Various masks and envelopes for handling different cutoff radii.

    Args:
        r_max (float): The maximum cutoff radius for interactions.
        equiv_r_max (float): The cutoff radius for equivariant features.
        basis_type (str): The type of radial basis function to use.
        radial_clamping_type (str): The type of clamping function for the radial basis.
        n_radial (int): The number of radial basis functions.
        n_invariant_features (int): The dimensionality of the latent features.
        irreps_edge_sh (o3.Irreps): The irreps for the spherical harmonics.
        n_equivariant_features (int): The multiplicity for the environment-weighted features.
        basis_kwargs (dict, optional): Additional keyword arguments for the basis function.
        n_mlp_layer (int): The number of layers in the MLPs.
        use_node_equivariant (bool): If True, computes initial node-centered equivariant
            features by scattering from the edges. Otherwise, computes edge-centered
            equivariant features.
        use_cuequivariance (bool): If True, use CUDA-accelerated implementations.
    """

    r_max: float

    def __init__(
        self,
        r_max: float,
        equiv_r_max: float,
        basis_type: Literal[
            "bessel", "gaussian", "smooth_bessel", "exp_norm", "mixture"
        ] = "bessel",
        radial_clamping_type: Literal["flat", "polynomial", "cosine"] = "flat",
        n_radial: int = 8,
        n_invariant_features: int = 128,
        irreps_edge_sh: o3.Irreps = o3.Irreps("1x0e+1x1o+1x2e"),
        n_equivariant_features: int = 32,
        basis_kwargs: Optional[dict[str, Any]] = None,
        n_mlp_layer: int = 1,
        use_node_equivariant: bool = False,
        use_cuequivariance: bool = False,
    ) -> None:
        if basis_kwargs is None:
            basis_kwargs = dict()
        super().__init__()

        self.use_node_equivariant = use_node_equivariant

        # Create an embedding layer for atomic numbers. This will be used for
        # both node and edge feature initialization.
        self.atomic_number_lin = AtomicEmbedding(embedding_dim=n_invariant_features)

        self.r_max = float(r_max)
        self.equiv_r_max = float(equiv_r_max)
        self.n_equivariant_features = n_equivariant_features

        # To avoid redundant initializations, we create a dictionary of radial basis
        # and clamping functions for each unique cutoff radius.
        r_max_list = sorted(list(set([r_max, equiv_r_max])))
        radial_basis_dict: dict[float, torch.nn.Module] = {}
        clamping_dict: dict[float, torch.nn.Module] = {}
        radial_basis_lin_dict: dict[float, torch.nn.Module] = {}
        for value in r_max_list:
            # Select the radial basis function type
            basis: torch.nn.Module
            if basis_type == "bessel":
                basis = BesselBasis(r_max=r_max, n_basis=n_radial, **basis_kwargs)
            elif basis_type == "gaussian":
                basis = GaussianBasis(r_max=r_max, n_basis=n_radial, **basis_kwargs)
            elif basis_type == "smooth_bessel":
                basis = SmoothBesselBasis(r_max=r_max, n_basis=n_radial)
            elif basis_type == "exp_norm":
                basis = ExpNormalSmearing(r_min=0.0, r_max=r_max, num_basis=n_radial)
            elif basis_type == "mixture":
                basis = MixExpNormSmoothBessel(
                    r_min=0.0,
                    r_max=r_max,
                    num_basis=n_radial,
                )
            else:
                raise ValueError(f"Unknown basis type: {basis_type}")
            radial_basis_dict[value] = basis

            # Select the radial clamping (envelope) function type
            if radial_clamping_type == "flat":
                clamping_dict[value] = StepClamp(r_max=value, offset=min(value / 3.0, 2.0))
            elif radial_clamping_type == "polynomial":
                clamping_dict[value] = PolynomialClamp(r_max=value)
            elif radial_clamping_type == "cosine":
                clamping_dict[value] = CosineClamp(r_max=value)
            else:
                raise ValueError(f"Unknown radial clamping type: {radial_clamping_type}")

            # Linear layer to project radial basis to the latent feature dimension
            radial_basis_lin_dict[value] = torch.nn.Linear(
                in_features=n_radial,
                out_features=n_invariant_features,
                bias=False,
            )

        # Assign the correct functions for the main interaction cutoff (r_max)
        self.radial_basis = radial_basis_dict[r_max]
        self.radial_clamping = clamping_dict[r_max]
        self.radial_basis_lin = radial_basis_lin_dict[r_max]

        # Assign the correct functions for the equivariant feature cutoff (equiv_r_max)
        self.radial_basis_equiv_r_max = radial_basis_dict[equiv_r_max]
        self.equiv_clamping = clamping_dict[equiv_r_max]
        self.radial_basis_lin_equiv_r_max = radial_basis_lin_dict[equiv_r_max]

        # --- Feature initialization settings ---
        self.n_sph_irreps = len(irreps_edge_sh)  # for "0e+1o" would be 2
        self.sph_dim = irreps_edge_sh.dim  # for "0e+1o" would be 4
        self.irreps_edge_feats = o3.Irreps(
            [(n_equivariant_features, ir) for _, ir in irreps_edge_sh]
        )

        # Linear layer to create an edge type embedding from the embeddings of the two atoms forming the edge.
        self.edge_type_lin = torch.nn.Linear(
            n_invariant_features * 2,
            n_invariant_features,
        )

        # MLP to compute initial invariant edge features from the radial basis
        # and the embeddings of the sender and receiver atoms.
        self.radial_basis_to_invariant_mlp = MLP(
            input_dim=n_radial + n_invariant_features * 2,
            output_dim=n_invariant_features,
            hidden_dims=[n_invariant_features] * n_mlp_layer,
            zero_bias=True,
        )

        # --- Equivariant feature settings ---
        self.sph_harmonics: torch.nn.Module
        self.use_cuequivariance = use_cuequivariance
        # Select the spherical harmonics implementation based on whether cuequivariance is used.
        if use_cuequivariance:
            self.sph_harmonics = cuequivariance_torch.SphericalHarmonics(
                ls=[ir.l for _, ir in irreps_edge_sh],
                normalize=True,
                device=torch.get_default_device(),
                math_dtype=torch.get_default_dtype(),
                use_fallback=False,
            )
        else:
            self.sph_harmonics = o3.SphericalHarmonics(
                irreps_edge_sh,
                normalize=True,
                normalization="component",
            )

        # MLP to compute the weights for the spherical harmonics from the radial basis
        # and the embeddings of the sender and receiver atoms.
        self.radial_basis_to_equivariant_mlp = MLP(
            input_dim=n_radial + n_invariant_features * 2,
            hidden_dims=[n_invariant_features] * n_mlp_layer,
            output_dim=self.irreps_edge_feats.num_irreps,
            zero_bias=True,
        )

        # Module to create weighted channels for the equivariant features.
        self.env_weighted: torch.nn.Module
        if use_cuequivariance:
            self.env_weighted = cue.MakeWeightedChannelsTwist(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )
        else:
            self.env_weighted = MakeWeightedChannels(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )

    # mocker for pickler
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    # mocker for pickler
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        receiver = data[keys.RECEIVER]
        sender = data[keys.SENDER]
        edge_length = data[keys.EDGE_LENGTH]

        # 1. Compute initial node features from atomic numbers
        # These are invariant features.
        atomic_number_lin_embed = self.atomic_number_lin(
            data[keys.ATOMIC_NUMBERS]
        )  # [n_nodes, n_invariant_features]

        # 2. Compute radial basis functions and envelopes
        # The envelope function ensures that features smoothly go to zero at the cutoff.
        envelope = self.radial_clamping(edge_length).unsqueeze(-1)

        # Project the radial basis functions to the latent feature dimension.
        radial_basis_embed = self.radial_basis(edge_length.unsqueeze(-1))  # [n_edges, n_radial]
        edge_length_embed = self.radial_basis_lin(
            radial_basis_embed
        )  # [n_edges, n_invariant_features]
        edge_length_embed = edge_length_embed * envelope  # Apply envelope

        # 3. Create masks and envelopes for different cutoff radii
        # This is necessary because equivariant features might use a smaller cutoff.
        if self.equiv_r_max == self.r_max:
            edge_mask = torch.ones_like(edge_length, dtype=torch.bool, device=edge_length.device)
            equiv_envelope = envelope
            radial_basis_embed_equiv_r_max = radial_basis_embed
            edge_length_embed_equiv_r_max = edge_length_embed
        else:
            edge_mask = edge_length < self.equiv_r_max
            equiv_envelope = self.equiv_clamping(edge_length[edge_mask]).unsqueeze(-1)
            radial_basis_embed_equiv_r_max = self.radial_basis_equiv_r_max(
                edge_length[edge_mask].unsqueeze(-1)
            )
            edge_length_embed_equiv_r_max = self.radial_basis_lin_equiv_r_max(
                radial_basis_embed_equiv_r_max
            )
            edge_length_embed_equiv_r_max = edge_length_embed_equiv_r_max * equiv_envelope

        # 4. Compute edge type embeddings
        # These are based on the atomic embeddings of the two atoms forming the edge.
        atomic_number_lin_rec = torch.index_select(atomic_number_lin_embed, 0, receiver)
        atomic_number_lin_sed = torch.index_select(atomic_number_lin_embed, 0, sender)
        edge_type_embed = self.edge_type_lin(
            torch.cat([atomic_number_lin_rec, atomic_number_lin_sed], dim=-1)
        )

        # 5. Compute spherical harmonics for the edge vectors (within the equivariant cutoff)
        # checkout model device
        n_equiv_edges = equiv_envelope.shape[0]
        # Note: collapse the data-dependent `n_equiv_edges == 0` branch
        # (originally a `.shape[0]` sync that broke CUDAGraph capture).
        # PR #3 + `tests/test_empty_equiv_mask.py` proved cueq's
        # `sph_harmonics` is safe on empty input; the branch was purely
        # defensive. The downstream `env_weighted` empty-fallback (~line 393)
        # is load-bearing and is intentionally NOT removed.
        torch._check(n_equiv_edges >= 0)
        spherical_harmonics = self.sph_harmonics(data[keys.EDGE_VECTORS][edge_mask])

        # 6. Compute initial invariant edge features
        # These are computed from the radial basis and the atomic embeddings of the sender and receiver.
        concatenated_type_radial = torch.concat(
            [radial_basis_embed, atomic_number_lin_rec, atomic_number_lin_sed], dim=-1
        )  # [ne, n_radial+n_features]

        edge_invariant = self.radial_basis_to_invariant_mlp(
            concatenated_type_radial
        )  # [ne, n_features]
        edge_invariant = edge_invariant * envelope  # [ne, n_features]

        # 7. Store all computed features in the data dictionary
        data[keys.EDGE_LENGTH_ENVELOPE] = envelope
        data[keys.EDGE_LENGTH_ENVELOPE_EQUI] = equiv_envelope
        data[keys.EDGE_MASK] = edge_mask
        data[keys.EDGE_EMBEDDING] = edge_length_embed  # [ne, n_features]
        data[keys.EDGE_EMBEDDING_EQUIV_RMAX] = edge_length_embed_equiv_r_max  # [ne, n_features]
        data[keys.EDGE_INVARIANT] = edge_invariant  # [ne, n_features]
        data[keys.EDGE_ATTRS] = spherical_harmonics  # [ne, n_irrep_dim]
        data[keys.EDGE_TYPE_EMBED] = edge_type_embed

        data[keys.NODE_INVARIANT] = atomic_number_lin_embed

        max_receiver_idx_t = data[keys.MAX_RECEIVER_IDX]
        # Note: under AOTI export, do NOT cast to a Python int.
        # `int(...)` forces specialization on the .item() result and the
        # exporter raises "could not extract specialized integer". Keep the
        # value as a SymInt and add `_check_is_size` + an upper bound so the
        # downstream `[:max_receiver_idx]` slice traces cleanly. Eager and
        # shipped torch.compile keep the int() cast for free.
        if is_aoti_export_mode():
            max_receiver_idx = max_receiver_idx_t.item()
            torch._check_is_size(max_receiver_idx)
            torch._check(max_receiver_idx <= atomic_number_lin_embed.shape[0])
        else:
            max_receiver_idx = int(max_receiver_idx_t.item())

        if keys.NLOCAL in data:
            # If the data contains a mask for local atoms, apply it to the node invariant features.
            data[keys.NODE_INVARIANT] = atomic_number_lin_embed[:max_receiver_idx]

        # 8. Compute initial equivariant features
        # First, compute the weights for the spherical harmonics.
        spherical_harmonics_weights = torch.concat(
            [
                radial_basis_embed_equiv_r_max,
                atomic_number_lin_rec[edge_mask],
                atomic_number_lin_sed[edge_mask],
            ],
            dim=-1,
        )  # [ne, n_radial+n_features]
        spherical_harmonics_weights = self.radial_basis_to_equivariant_mlp(
            spherical_harmonics_weights
        )  # [ne, n_sph_irreps]

        # The weights must also be enveloped to ensure they go to zero at the cutoff.
        spherical_harmonics_weights = (
            spherical_harmonics_weights * equiv_envelope
        )  # [ne, n_sph_irreps]

        # Weight the spherical harmonics to get the initial equivariant edge
        # features. Note: this guard is *load-bearing* in eager
        # (cueq's MakeWeightedChannelsTwist crashes on empty input;
        # `tests/test_empty_equiv_mask.py::test_..._cueq_env_weighted_crashes_on_empty`),
        # so we keep the empty fallback in eager / shipped torch.compile but
        # collapse to the always-take path under the AOTI-export flag. The
        # mliap.py AOTI dispatcher is responsible for routing stretched
        # (n_equiv_edges == 0) batches to eager so the export-friendly path's
        # `torch._check(n_equiv_edges > 1)` invariant holds at runtime.
        if is_aoti_export_mode():
            torch._check(n_equiv_edges > 1)
            edge_equivariant = self.env_weighted(spherical_harmonics, spherical_harmonics_weights)
        elif n_equiv_edges == 0:
            if self.use_cuequivariance:
                edge_equivariant = torch.empty(
                    (0, self.sph_dim, self.n_equivariant_features),
                    device=edge_length.device,
                )
            else:
                edge_equivariant = torch.empty(
                    (0, self.n_equivariant_features, self.sph_dim),
                    device=edge_length.device,
                )
        else:
            edge_equivariant = self.env_weighted(spherical_harmonics, spherical_harmonics_weights)
        # note, the shape would be [n_edges, sph_dim, n_equivariant_features] if use cuequivariance
        # otherwise, it would be [n_edges, n_equivariant_features, sph_dim]

        # 9. Optionally, scatter edge equivariant features to nodes
        if self.use_node_equivariant:
            node_equivariant = scatter_sum(
                src=edge_equivariant,
                index=receiver[edge_mask],
                dim=0,
                dim_size=max_receiver_idx,
            )
            data[keys.NODE_EQUIVARIANT] = node_equivariant
        else:
            data[keys.EDGE_EQUIVARIANT] = edge_equivariant

        # remove per-atom energy if exists
        if keys.PER_ATOM_ENERGY in data:
            del data[keys.PER_ATOM_ENERGY]

        return data


# class GATLocalAttention(torch.nn.Module) was deleted


class InvariantSelfAttention(torch.nn.Module):
    def __init__(
        self,
        n_invariant_features: int = 128,
        n_attn_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        n_mlp_layer: int = 1,
        attention_type: str = "exp",
    ) -> None:
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(
            n_invariant_features,
            elementwise_affine=False,
            eps=1e-5,
        )

        self.n_attn_heads = n_attn_heads
        if attn_head_dim is None:
            if n_invariant_features % n_attn_heads == 0:
                attn_head_dim = n_invariant_features // n_attn_heads
            else:
                attn_head_dim = max(8, int(n_invariant_features / n_attn_heads) * 2)
        self.attn_head_dim = int(attn_head_dim)
        self.scaling_factor = 1 / math.sqrt(self.attn_head_dim)

        self.n_invariant_features = n_invariant_features

        self.attention_type: torch.nn.Module
        transform_types = {
            "softmax": QKVSoftmax,
            "exp": QKVExp,
            "silu": QKVSiLu,
            "concat": QKVConcat,
        }
        self.attention_type_node = transform_types[attention_type](
            num_attn_heads=n_attn_heads,
            attn_head_dim=attn_head_dim,
            eps=1.0,
        )
        self.edge_type_lin = torch.nn.Linear(
            in_features=n_invariant_features, out_features=n_invariant_features, bias=False
        )

        # edge query, node key, node value --> new edge invariant
        hidden_dim = 256 if n_invariant_features < 256 else n_invariant_features
        self.edge_mlp = NoBiasMLP(
            # input_dim=self.n_attn_heads * self.attn_head_dim,
            input_dim=n_invariant_features,
            output_dim=n_invariant_features,
            hidden_dims=[hidden_dim] * n_mlp_layer,
        )

        # filter needs no bias because the embedding vanish to zero when close to cutoff
        self.edge_filter_mlp = NoBiasMLP(
            input_dim=n_invariant_features,
            output_dim=self.n_attn_heads * self.attn_head_dim,
            hidden_dims=[n_invariant_features] * n_mlp_layer,
        )

        # node attention part
        # all linear components do not have bias
        linear_kwargs = dict(
            in_features=n_invariant_features,
            out_features=self.n_attn_heads * self.attn_head_dim,
        )
        self.node_query_lin = torch.nn.Linear(**linear_kwargs, bias=False)
        self.edge_key_lin = torch.nn.Linear(**linear_kwargs, bias=False)
        self.edge_value_lin = torch.nn.Linear(**linear_kwargs, bias=False)
        self.node_out_lin = torch.nn.Linear(
            in_features=self.n_attn_heads * self.attn_head_dim,
            # n_invariant_features,
            out_features=n_invariant_features,
            bias=False,
        )
        self.post_attn_mlp_node = MLP(
            # input_dim=self.n_attn_heads * self.attn_head_dim,
            input_dim=n_invariant_features,
            output_dim=n_invariant_features,
            hidden_dims=[hidden_dim] * n_mlp_layer,
            zero_bias=True,
        )

    def forward(
        self, data: dict[str, torch.Tensor], trace: bool = False
    ) -> dict[str, torch.Tensor]:

        # Note: removed the `if receiver.numel() == 0: return data` early-out.
        # `.numel()` triggers a graph break under torch.compile and blocks
        # CUDAGraph capture. The downstream attention + scatter_sum are safe on
        # 0-edge inputs (linear ops on empty tensors are no-ops; scatter_sum
        # returns a zero-filled tensor of shape `[max_receiver_idx, ...]`).
        receiver = data[keys.RECEIVER]
        torch._check(receiver.numel() >= 0)

        envelope = data[keys.EDGE_LENGTH_ENVELOPE]
        node_invariant = data[keys.NODE_INVARIANT]
        edge_invariant = data[keys.EDGE_INVARIANT]
        edge_length_embed = data[keys.EDGE_EMBEDDING]
        edge_type_embed = data[keys.EDGE_TYPE_EMBED]

        key_ij = self.edge_type_lin(edge_type_embed)

        # could use concatenation or multiplication instead
        new_edge_invariant = self.edge_mlp(edge_invariant)
        # new_edge_invariant = new_edge_invariant * key_ij
        # new_edge_invariant = edge_invariant + new_edge_invariant
        new_edge_invariant = torch.addcmul(edge_invariant, key_ij, new_edge_invariant)

        # we do not multiply by envelope here as envelope is only needed when there is a scatter sum
        # (like to compute node invariant or to compute energy)
        data[keys.EDGE_INVARIANT] = new_edge_invariant

        # attention block
        # with node query, edge key, and edge value
        query_i = self.node_query_lin(node_invariant)  # [node, n_heads * head_dim]
        query_i = query_i.index_select(0, receiver)  # [edge, n_heads * head_dim]
        key_ij = self.edge_key_lin(new_edge_invariant)  # [edge, n_heads * head_dim]

        # watch out for the filter operation. is plus better than multiplication?
        edge_filter = self.edge_filter_mlp(edge_length_embed)
        key_ij = key_ij + edge_filter  # [edge, n_heads * head_dim]
        value_ij = self.edge_value_lin(new_edge_invariant)
        value_ij = value_ij + edge_filter

        # remember to keep cutoff here, so close to cutoff edges contribute zero
        edge_invariant_to_sum = self.attention_type_node(
            receiver,
            query_i,
            key_ij,
            value_ij,
            envelope,
            dim_size=node_invariant.size(0),
        )

        new_node_invariant = scatter_sum(
            src=edge_invariant_to_sum, index=receiver, dim=0, dim_size=node_invariant.size(0)
        )

        new_node_invariant = self.node_out_lin(new_node_invariant)
        new_node_invariant = node_invariant + new_node_invariant
        new_node_invariant = self.layer_norm(new_node_invariant)
        new_node_invariant = self.post_attn_mlp_node(new_node_invariant)
        new_node_invariant = node_invariant + new_node_invariant

        data[keys.NODE_INVARIANT] = new_node_invariant

        return data


class EnergyOutput(torch.nn.Module):
    def __init__(
        self,
        e0s: Optional[torch.Tensor] = None,
        n_invariant_features: int = 128,
        mlp_kwargs: Optional[dict[str, Any]] = None,
        edge_energy_output: bool = True,
        node_energy_output: bool = False,
        out_scale: float = 1.0,
        trainable_e0: bool = False,
        out_shift: float = 0.0,
    ) -> None:

        if mlp_kwargs is None:
            mlp_kwargs = dict(hidden_features=[128])

        super().__init__()
        self.edge_energy_output = edge_energy_output
        self.node_energy_output = node_energy_output
        if node_energy_output:
            self.node_eng_mlp: torch.nn.Module = NoBiasMLP(
                input_dim=n_invariant_features,
                output_dim=1,
                **mlp_kwargs,
            )
            with torch.no_grad():
                self.node_eng_mlp[-1].weight *= out_scale
        else:
            self.node_eng_mlp = torch.nn.Identity()

        if edge_energy_output:
            self.edge_eng_mlp: torch.nn.Module = NoBiasMLP(
                input_dim=n_invariant_features,
                output_dim=1,
                **mlp_kwargs,
            )
            with torch.no_grad():
                self.edge_eng_mlp[-1].weight *= out_scale * 0.01
            self.edge_energy_output = edge_energy_output
        else:
            self.edge_eng_mlp = torch.nn.Identity()

        _e0s = torch.zeros((constant.MAX_ATOMIC_NUMBER + 1, 1)) if e0s is None else e0s
        _e0s = _e0s + out_shift
        if trainable_e0:
            self.e0s = torch.nn.Parameter(_e0s)
        else:
            self.register_buffer("e0s", _e0s)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        max_receiver_idx = data[keys.NODE_INVARIANT].size(0)
        atomic_numbers = data[keys.ATOMIC_NUMBERS][:max_receiver_idx]
        per_atom_eng = self.e0s.index_select(0, atomic_numbers)

        # MLP from edge latent to edge energy
        if self.node_energy_output:
            node_invariant = data[keys.NODE_INVARIANT]
            per_atom_eng = per_atom_eng + (self.node_eng_mlp(node_invariant)).view(
                per_atom_eng.shape
            )

        if self.edge_energy_output:
            # Note: removed the `if receiver.numel() > 0:` gate. `.numel()`
            # is a data-dependent guard that breaks CUDAGraph capture. The
            # MLP and scatter_sum are safe on 0-edge inputs; the latter
            # returns a zero contribution which is the correct identity for
            # an additive accumulation into per_atom_eng.
            receiver = data[keys.RECEIVER]
            envelope = data[keys.EDGE_LENGTH_ENVELOPE]
            torch._check(receiver.numel() >= 0)
            edge_invariant = data[keys.EDGE_INVARIANT] * envelope
            edge_eng = self.edge_eng_mlp(edge_invariant)
            data[keys.EDGE_ENERGY] = edge_eng
            per_atom_eng = per_atom_eng + (
                scatter_sum(
                    src=edge_eng,
                    index=receiver,
                    dim=0,
                    dim_size=max_receiver_idx,
                )
            ).view(per_atom_eng.shape)

        return {
            keys.PER_ATOM_ENERGY: per_atom_eng,
        }
