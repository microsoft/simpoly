import logging
from typing import Any, Final, Literal, Optional

import torch
from e3nn import o3

from simpoly.vivace import data, keys, utils
from simpoly.vivace.models.base import MLFFClassMetadata, MLFFInstanceMetadata, MLFFModel
from simpoly.vivace.utils import scatter

from .equivariant import (
    InvariantCrossAttention2EquivariantEdge,
    InvariantCrossAttention2EquivariantNode,
)
from .layers import EnergyOutput, InitHeader, InvariantSelfAttention
from .utils import build_irreps

O3_RESTRICTED: Final[str] = "o3_restricted"
PARITY_SETTING_TYPES = Literal["o3_full", "o3_restricted", "so3"]
O3_FULL: Final[PARITY_SETTING_TYPES] = "o3_full"
SO3: Final[PARITY_SETTING_TYPES] = "so3"
CHEMICAL_ENCODING_TYPES = Literal["onehot", "estructure", "orb_proj"]
ONEHOT: Final[CHEMICAL_ENCODING_TYPES] = "onehot"
ESTRUCTURE: Final[CHEMICAL_ENCODING_TYPES] = "estructure"
ORB_PROJ: Final[CHEMICAL_ENCODING_TYPES] = "orb_proj"

LOG = logging.getLogger(__name__)


class Vivace(MLFFModel):
    """
    Vivace: A SO(3)/O(3) equivariant graph neural network model for molecular dynamics.

    This model uses a combination of invariant and equivariant features, processed
    through a series of attention-based layers, to predict energies, forces, and
    other properties of molecular systems.

    The architecture consists of three main parts:
    1. An `InitHeader` that computes initial node and edge features from atomic
       positions and types.
    2. A series of attention blocks that update these features. This includes
       invariant self-attention and invariant-to-equivariant cross-attention.
    3. An `EnergyOutput` layer that predicts the total energy from the final features.

    Args:
        r_max (float): The maximum cutoff radius for interactions.
        l_max (int): The maximum degree of spherical harmonics to use.
        e0s (torch.Tensor): A tensor of reference atomic energies.
        n_layers (int): The number of equivariant cross-attention layers.
        n_invariant_pre_layers (int): Number of invariant self-attention layers
            to apply before the main equivariant blocks.
        n_invariant_sub_layers (int): Number of invariant self-attention layers
            to apply within each main equivariant block.
        n_invariant_pos_layers (int): Number of invariant self-attention layers
            to apply after the main equivariant blocks.
        n_radial (int): The number of radial basis functions.
        n_invariant_features (int): The dimensionality of the latent invariant features.
        n_equivariant_features (int, optional): The multiplicity for the environment-weighted
            equivariant features. Defaults to `n_invariant_features // 2`.
        n_attn_heads (int, optional): Number of attention heads for invariant attention.
            Defaults to `n_invariant_features // 16`.
        attn_head_dim (int, optional): Dimensionality of each invariant attention head.
        out_scale (float): A scaling factor for the output energy.
        out_shift (float): A shift for the output energy.
        basis_kwargs (dict, optional): Additional keyword arguments for the basis function.
        n_atomic_type_mlp_layer (int): Number of layers in the MLP for atomic type embeddings.
        n_post_attn_mlp_layer (int): Number of layers in the MLP after attention.
        eng_mlp_kwargs (dict, optional): Keyword arguments for the energy prediction MLP.
        equiv_r_max (float, optional): The cutoff radius for equivariant features.
            Defaults to `r_max`.
        equiv_n_attn_heads (int, optional): Number of attention heads for equivariant
            attention. Defaults to `n_attn_heads`.
        equiv_attn_head_dim (int, optional): Dimensionality of each equivariant attention head.
        parity (str): The parity setting for spherical harmonics ('o3_full', 'o3_restricted', 'so3').
        basis_type (str): The type of radial basis function.
        radial_clamping_type (str): The type of clamping function for the radial basis.
        attention_type (str): The transformation for attention queries, keys, and values.
        use_cuequivariance (bool): If True, use CUDA-accelerated implementations.
        use_all_tp_paths (bool): If True, use all possible tensor product paths.
        use_node_equivariant (bool): If True, the model is node-centered. Otherwise, it is
            edge-centered.
    """

    force_to_use_node_equivariant: bool = False

    def __init__(
        self,
        r_max: float,
        l_max: int,
        e0s: torch.Tensor,
        # stacking
        n_layers: int,
        n_invariant_pre_layers: int = 0,
        n_invariant_sub_layers: int = 1,
        n_invariant_pos_layers: int = 0,
        # features
        n_radial: int = 8,
        n_invariant_features: int = 128,
        n_equivariant_features: Optional[int] = None,
        # attention
        n_attn_heads: Optional[int] = None,
        attn_head_dim: Optional[int] = None,
        out_scale: float = 0.1,
        out_shift: float = 0.0,
        basis_kwargs: Optional[dict[str, Any]] = None,
        # MLP & linear
        n_atomic_type_mlp_layer: int = 1,
        n_post_attn_mlp_layer: int = 1,
        eng_mlp_kwargs: Optional[dict[str, Any]] = None,
        # equivariant part can be set differently. Optional
        equiv_r_max: Optional[float] = None,
        equiv_n_attn_heads: Optional[int] = None,
        equiv_attn_head_dim: Optional[int] = None,
        # architecture details
        parity: PARITY_SETTING_TYPES = O3_FULL,
        basis_type: Literal["bessel", "gaussian", "mixture", "exp_norm"] = "gaussian",
        radial_clamping_type: Literal["flat", "polynomial", "cosine"] = "flat",
        attention_type: Literal["exp", "concat", "silu", "softmax"] = "concat",
        use_cuequivariance: bool = torch.cuda.is_available(),  # cuequivariance require cuda to run
        use_node_equivariant: bool = False,
        use_all_tp_paths: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:

        deprecated_arguments = [
            "equi_r_max",
            "cutoff_type",
            "env_embed_multiplicity",
            "two_body_latent_mlp_latent_dimensions",
            "latent_mlp_latent_dimensions",
            "latent_resnet",
            "edge_eng_mlp_latent_dimensions",
            "edge_eng_mlp_nonlinearity",
            "num_basis",
            "num_layers",
            "qkv_transform",
            "env_multiplicity",
            "num_invariant_pre_layers",
            "num_invariant_sub_layers",
            "num_invariant_pos_layers",
            "num_attn_heads",
            "equiv_num_attn_heads",
            "num_latent_features",
            "n_channels",
        ]
        for k, v in kwargs.items():
            LOG.warning(f"Unused argument {k}={v}")
            if k.startswith("equi_"):
                raise ValueError(f"Argument {k}={v} is not supported. Use equiv_{k[5:]} instead.")
            elif k == "cutoff_type":
                raise ValueError(
                    f"Argument {k}={v} is not supported. Use radial_clamping_type instead."
                )
            elif k == "env_embed_multiplicity":
                raise ValueError(
                    f"Argument {k}={v} is not supported. Use n_equivariant_features instead."
                )
            for dep_arg in deprecated_arguments:
                if k == dep_arg:
                    raise ValueError(f"Argument {k}={v} is not supported")

        if self.force_to_use_node_equivariant and not use_node_equivariant:
            use_node_equivariant = True
            LOG.warning(
                "force_to_use_node_equivariant is set to True, forcing use_node_equivariant=True"
            )

        if basis_kwargs is None:
            basis_kwargs = dict()

        if n_equivariant_features is None:
            n_equivariant_features = n_invariant_features // 2
        if n_attn_heads is None:
            n_attn_heads = n_invariant_features // 16 if n_invariant_features >= 64 else 4

        if equiv_n_attn_heads is None:
            equiv_n_attn_heads = n_attn_heads
        if equiv_attn_head_dim is None:
            equiv_attn_head_dim = attn_head_dim

        if eng_mlp_kwargs is None:
            eng_mlp_kwargs = dict(hidden_dims=[64])

        super().__init__()

        equiv_r_max = r_max if equiv_r_max is None else equiv_r_max
        assert equiv_r_max <= r_max, "equiv_r_max must be less than or equal to r_max"

        # create irreps for all the TP in the network
        irreps_edge_sh = o3.Irreps.spherical_harmonics(l_max, p=(1 if parity == SO3 else -1))
        self.init_features: torch.nn.Module = InitHeader(
            r_max=r_max,
            equiv_r_max=equiv_r_max,
            basis_type=basis_type,
            radial_clamping_type=radial_clamping_type,
            n_radial=n_radial,
            n_invariant_features=n_invariant_features,
            irreps_edge_sh=irreps_edge_sh,
            n_equivariant_features=n_equivariant_features,
            basis_kwargs=basis_kwargs,
            n_mlp_layer=n_atomic_type_mlp_layer,
            use_cuequivariance=use_cuequivariance,
            use_node_equivariant=use_node_equivariant,
        )

        self.invariant_action_blocks = torch.nn.ModuleList()
        irreps_in = self.init_features.irreps_edge_feats
        self_attention: torch.nn.Module
        for layer_idx in range(n_invariant_pre_layers):
            self_attention = InvariantSelfAttention(
                n_invariant_features=n_invariant_features,
                n_attn_heads=n_attn_heads,
                attn_head_dim=attn_head_dim,
                attention_type=attention_type,
                n_mlp_layer=n_post_attn_mlp_layer,
            )
            self.invariant_action_blocks.append(self_attention)

        self.use_node_equivariant = use_node_equivariant
        cross_attention_class = (
            InvariantCrossAttention2EquivariantNode
            if use_node_equivariant
            else InvariantCrossAttention2EquivariantEdge
        )

        self.action_blocks = torch.nn.ModuleList()
        if n_layers > 0:
            _, tps_irreps_out = build_irreps(
                input_irreps=o3.Irreps(irreps_edge_sh),
                hidden_irreps=o3.Irreps(irreps_edge_sh),
                final_irreps=o3.Irreps([(1, (0, 1))]),
                n_layers=n_layers,
                nonscalars_include_parity=parity == O3_FULL,
                use_all_tp_paths=use_all_tp_paths,
            )
            for layer_idx in range(n_layers):
                irreps_out = o3.Irreps(
                    [(n_equivariant_features, ir) for _, ir in tps_irreps_out[layer_idx]]
                )
                equiv_attention = cross_attention_class(
                    irreps_in=irreps_in,
                    irreps_out=irreps_out,
                    irreps_edge_sh=irreps_edge_sh,
                    n_invariant_features=n_invariant_features,
                    n_equivariant_features=n_equivariant_features,
                    n_attn_heads=equiv_n_attn_heads,
                    attn_head_dim=equiv_attn_head_dim,
                    use_l0_only=not use_all_tp_paths,
                    use_cuequivariance=use_cuequivariance,
                    attention_type=attention_type,
                    n_mlp_layer=n_post_attn_mlp_layer,
                )
                self.action_blocks.append(equiv_attention)
                for _ in range(n_invariant_sub_layers):
                    self_attention = InvariantSelfAttention(
                        n_invariant_features=n_invariant_features,
                        n_attn_heads=n_attn_heads,
                        attn_head_dim=attn_head_dim,
                        attention_type=attention_type,
                        n_mlp_layer=n_post_attn_mlp_layer,
                    )
                    self.action_blocks.append(self_attention)
                irreps_in = irreps_out
        for layer_idx in range(n_invariant_pos_layers):
            self_attention = InvariantSelfAttention(
                n_invariant_features=n_invariant_features,
                n_attn_heads=n_attn_heads,
                attn_head_dim=attn_head_dim,
                attention_type=attention_type,
                n_mlp_layer=n_post_attn_mlp_layer,
            )
            self.action_blocks.append(self_attention)

        self.eng_output = EnergyOutput(
            n_invariant_features=n_invariant_features,
            mlp_kwargs=eng_mlp_kwargs,
            edge_energy_output=True,
            node_energy_output=True,
            out_scale=out_scale,
            trainable_e0=False,
            e0s=e0s,
            out_shift=out_shift,
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.n_invariant_features = n_invariant_features

    def get_instance_metadata(self) -> MLFFInstanceMetadata:
        return MLFFInstanceMetadata(
            r_max=self.r_max.item(),
            lammps_cutoff=self.r_max.item(),
            dtype=self.r_max.dtype,
        )

    @classmethod
    def get_cls_metadata(cls) -> MLFFClassMetadata:
        return MLFFClassMetadata(
            is_edge_centered=True,
        )

    def get_latent(
        self,
        batch: dict[str, torch.Tensor],
        trace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """assuming edge index and all related data are already computed, return all the latent variables"""

        output = {k: v for k, v in batch.items()}
        output = self.init_features(output)

        node_invariant = output[keys.NODE_INVARIANT]
        edge_invariant = output[keys.EDGE_INVARIANT]
        edge_envelope = output[keys.EDGE_LENGTH_ENVELOPE]
        # equivariant_key = keys.NODE_EQUIVARIANT
        # equivariant = output[equivariant_key]

        # if self.remove_simple_elements:
        # output = remove_edges_with_specific_receivers(output, self.atomic_numbers_to_remove)
        equivariant_key = (
            keys.NODE_EQUIVARIANT if self.use_node_equivariant else keys.EDGE_EQUIVARIANT
        )

        for action_block in self.invariant_action_blocks:
            # output = action_block(output, trace=trace)
            output = action_block(output)

        for action_block in self.action_blocks:
            output = action_block(output)  # , trace=trace)

        node_invariant = output[keys.NODE_INVARIANT]
        edge_invariant = output[keys.EDGE_INVARIANT]
        # equivariant = output[equivariant_key]

        result = {
            keys.RECEIVER: batch[keys.RECEIVER],
            keys.SENDER: batch[keys.SENDER],
            keys.ATOMIC_NUMBERS: batch[keys.ATOMIC_NUMBERS],
            keys.EDGE_LENGTH_ENVELOPE: edge_envelope,
            keys.EDGE_INVARIANT: edge_invariant,
            keys.NODE_INVARIANT: node_invariant,
        }
        for key in [keys.BATCH, keys.BATCH_PTR]:
            if key in batch:
                result[key] = batch[key]
        result[equivariant_key] = output[equivariant_key]
        return result

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = True,
        trace: bool = False,
        mode: str = keys.BATCH_MODE,
    ) -> dict[str, Optional[torch.Tensor]]:

        if mode == keys.BATCH_MODE:
            edge_data = data.compute_pair_edge_data_batch(
                data=batch,
                compute_grads=compute_forces or compute_virial,
            )
        else:
            edge_data = data.compute_pair_edge_data_lammps(data=batch)

        batch.update(edge_data)

        output = {k: v for k, v in batch.items()}

        # with utils.TimerConetextManager("GetLatent"):
        latent = self.get_latent(output)

        # with utils.TimerConetextManager("EngOutput"):

        output = self.eng_output(latent)
        e_i = output[keys.PER_ATOM_ENERGY]
        e_i = e_i.view([-1])

        if mode == keys.BATCH_MODE:
            # Note: hoist the `.item()` out of the LAMMPS forward path.
            # `max_receiver_idx` is only consumed inside the BATCH_MODE branch
            # (passed to compute_pair_properties_batch as `n_nodes`); the
            # LAMMPS path uses compute_pair_properties_lammps which does not
            # need it. Moving the `.item()` here removes one CPU↔GPU sync
            # from the production MLIAP forward (CUDAGraph prerequisite).
            max_receiver_idx = int(batch[keys.MAX_RECEIVER_IDX].item())
            n_graphs = batch[keys.BATCH_PTR].shape[0] - 1
            if n_graphs > 1:
                energy = scatter.scatter_sum(  # [n_graphsc
                    src=e_i,
                    index=batch[keys.BATCH],
                    dim=0,
                    dim_size=n_graphs,
                )
            else:
                energy = torch.sum(e_i).view([-1])

            props = utils.compute_pair_properties_batch(
                energy=energy,
                vectors=batch[keys.EDGE_VECTORS],
                receiver=batch[keys.RECEIVER],
                sender=batch[keys.SENDER],
                n_graphs=n_graphs,
                n_nodes=max_receiver_idx,  # n_nodes is the max_receiver_index
                n_edges_per_graph=batch[keys.N_EDGES_PER_GRAPH],
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                training=self.training,
            )

        else:
            energy = torch.sum(e_i, dim=0)
            props = utils.compute_pair_properties_lammps(
                energy, batch[keys.EDGE_VECTORS], batch[keys.REVERSE_DIST_SORTING_IDX]
            )
        props[keys.PER_ATOM_ENERGY] = e_i

        return props

    def __getstate__(self) -> dict[Any, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__ = state
