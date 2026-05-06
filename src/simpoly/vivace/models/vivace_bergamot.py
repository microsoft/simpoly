"""
This is a very experimental test case, where we try to add some strong short range repulsion (mimicing ZBL potential)
and an atomic charge prediction head to help with generalization.
"""

import logging
from typing import Any, Optional

import torch

from simpoly.vivace import constant, data, keys, utils
from simpoly.vivace.models.base import (
    MLFFClassMetadata,
    MLFFInstanceMetadata,
    MLFFModel,
    MLFFMultiHeadedModel,
)
from simpoly.vivace.modules import MLP, CosineClamp
from simpoly.vivace.utils import scatter

from .layers import EnergyOutput
from .layers import is_aoti_export_mode as _aoti_export_mode
from .vivace import O3_FULL, Vivace

LOG = logging.getLogger(__name__)


class VivaceCitron(MLFFMultiHeadedModel, Vivace):
    allowed_heads: list[str] = ["cp2k", "orca", "xtb", "vasp", "omol"]

    force_to_use_node_equivariant: bool = False

    def __init__(
        self,
        r_max: float,
        e0s: dict[str, torch.Tensor],
        eng_mlp_kwargs: dict[str, Any],
        out_scale: float = 1.0,
        out_shift: float = 0.0,
        use_positive_short_range: bool = True,
        short_range_r_max: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        use_positive_short_range: whether to use positive short range repulsion or allow the pre-factor to be negative
        short_range_r_max: the cutoff for the short range explosion term
        """

        MLFFMultiHeadedModel.__init__(self)

        # get an backbone model
        place_holder_e0s = torch.zeros(
            constant.MAX_ATOMIC_NUMBER, 1, dtype=torch.get_default_dtype()
        )
        Vivace.__init__(self, r_max=r_max, e0s=place_holder_e0s, **kwargs)
        assert hasattr(self, "n_invariant_features")
        assert hasattr(self, "get_latent")
        assert hasattr(self, "eng_output")
        del self.eng_output
        self.use_positive_short_range = use_positive_short_range

        self.register_buffer(
            "short_range_r_max", torch.as_tensor(short_range_r_max, dtype=torch.get_default_dtype())
        )
        self.n_r_power = 12

        self.cosine_clamp = CosineClamp(r_max=short_range_r_max)
        self.short_range_mlp = MLP(
            input_dim=self.n_invariant_features,
            hidden_dims=[self.n_invariant_features] * 2,
            output_dim=self.n_r_power,
        )

        self.charge_mlp = MLP(
            input_dim=self.n_invariant_features,
            hidden_dims=[self.n_invariant_features] * 2,
            output_dim=1,
        )

        self.default_head = "cp2k"
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))

        # add the heads
        for head in self.allowed_heads:
            head_module: torch.nn.Module = EnergyOutput(
                n_invariant_features=self.n_invariant_features,
                mlp_kwargs=eng_mlp_kwargs,
                out_scale=out_scale,
                edge_energy_output=True,
                node_energy_output=True,
                e0s=e0s[head],
                trainable_e0=False,
                out_shift=out_shift,
            )
            self.add_module(f"_head_{head}", head_module)
        LOG.info(f"Default head: {self.default_head}")

    def short_range_explosion(
        self,
        edge_length: torch.Tensor,
        edge_feature: torch.Tensor,
        receiver: torch.Tensor,
        max_receiver_idx: int,
    ) -> torch.Tensor:
        is_short_edge = edge_length < self.short_range_r_max
        # Note: under AOTI export the `if is_short_edge.any():` branch
        # is data-dependent. Force the always-take path under export only;
        # eager + shipped torch.compile keep the original branch.
        if _aoti_export_mode() or is_short_edge.any():
            short_edge_feature = edge_feature[
                is_short_edge
            ]  # [n_short_edges, n_invariant_features]
            r = edge_length[is_short_edge]  # [n_short_edges, 1]

            # use square to make sure it is repulsion
            if self.use_positive_short_range:
                w_n = torch.square(self.short_range_mlp(short_edge_feature))  # [n_short_edges, 5]
            else:
                w_n = self.short_range_mlp(short_edge_feature)  # [n_short_edges, 5]

            cosine_clamp = self.cosine_clamp(edge_length[is_short_edge]).unsqueeze(
                -1
            )  # [n_short_edges, 1]
            powers = torch.arange(1, self.n_r_power + 1, device=r.device, dtype=r.dtype).unsqueeze(
                0
            )  # [1, n_n_r_power]
            inv_r_pow = r.unsqueeze(-1).pow(-powers)  # [n_short_edges, n_n_r_power]
            explosion = (w_n * inv_r_pow).sum(
                dim=-1, keepdim=True
            ) * cosine_clamp  # [n_short_edges, 1]

            receiver = receiver[is_short_edge]
            per_atom_energy = scatter.scatter_sum(
                src=explosion,
                index=receiver,
                dim=0,
                dim_size=max_receiver_idx,
            )
            return per_atom_energy.squeeze(-1)  # [max_receiver_idx]
        return torch.zeros(max_receiver_idx, dtype=edge_feature.dtype, device=edge_feature.device)

    def from_latent_to_energy(
        self,
        latent: dict[str, torch.Tensor],
        head_id: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        head_str = "orca" if head_id is None else head_id
        assert head_str in ["omol", "orca", "xtb", "vasp", "cp2k"], f"Unknown head: {head_str}"
        if head_str == "orca":
            return self._head_orca(latent)  # type: ignore[no-any-return]
        elif head_str == "xtb":
            return self._head_xtb(latent)  # type: ignore[no-any-return]
        elif head_str == "omol":
            return self._head_omol(latent)  # type: ignore[no-any-return]
        elif head_str == "vasp":
            return self._head_vasp(latent)  # type: ignore[no-any-return]
        return self._head_cp2k(latent)  # type: ignore[no-any-return]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        head_id: Optional[str] = None,  # type: ignore[override]
        compute_forces: bool = True,
        compute_virial: bool = True,
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

        latent = self.get_latent(batch)

        output = self.from_latent_to_energy(
            latent=latent,
            head_id=head_id,
        )

        # with utils.TimerConetextManager("EngOutput"):
        # Note: under AOTI export the `int(t.item())` cast forces a
        # specialized integer; drop the cast and pass the SymInt through.
        if _aoti_export_mode():
            _mri_t = batch[keys.MAX_RECEIVER_IDX]
            _mri_sym = _mri_t.item()
            torch._check(_mri_sym >= 0)
            max_receiver_idx_arg: int = _mri_sym  # type: ignore[assignment]
        else:
            max_receiver_idx_arg = int(batch[keys.MAX_RECEIVER_IDX].item())
        short_range_per_atom_energy = self.short_range_explosion(
            edge_feature=latent[keys.EDGE_INVARIANT],
            edge_length=batch[keys.EDGE_LENGTH],
            receiver=batch[keys.RECEIVER],
            max_receiver_idx=max_receiver_idx_arg,
        ).view([-1])

        e_i = output[keys.PER_ATOM_ENERGY]
        assert e_i is not None, "Energy output is None"  # pleasing mypy
        e_i = e_i.view([-1]) + short_range_per_atom_energy.view([-1])

        output[keys.PER_ATOM_ENERGY] = e_i

        if _aoti_export_mode():
            max_receiver_idx = max_receiver_idx_arg
        else:
            max_receiver_idx = int(batch[keys.MAX_RECEIVER_IDX].item())

        if mode == keys.BATCH_MODE:
            n_graphs = batch[keys.BATCH_PTR].shape[0] - 1
            if n_graphs > 1:
                energy = scatter.scatter_sum(  # [n_graphsc
                    src=e_i,
                    index=batch[keys.BATCH],
                    dim=0,
                    dim_size=n_graphs,
                )
            else:
                energy = torch.sum(e_i).reshape([-1])

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

        charge = self.charge_mlp(latent[keys.NODE_INVARIANT]).view([-1])
        props[keys.MULLIKEN_ATOMIC_CHARGES] = charge

        return props

    def get_instance_metadata(self) -> MLFFInstanceMetadata:
        r_max = self.r_max.item()
        return MLFFInstanceMetadata(
            r_max=r_max,
            lammps_cutoff=r_max,
            dtype=self.r_max.dtype,
        )

    @classmethod
    def get_cls_metadata(cls) -> MLFFClassMetadata:
        return MLFFClassMetadata(
            is_edge_centered=True,
        )


class VivaceBergamot(VivaceCitron):
    allowed_heads: list[str] = ["cp2k", "orca", "xtb", "omol"]

    def from_latent_to_energy(
        self,
        latent: dict[str, torch.Tensor],
        head_id: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        head_str = "cp2k" if head_id is None else head_id
        assert head_str in ["cp2k", "orca", "xtb", "omol"], f"Unknown head: {head_str}"
        if head_str == "orca":
            return self._head_orca(latent)  # type: ignore[no-any-return]
        elif head_str == "xtb":
            return self._head_xtb(latent)  # type: ignore[no-any-return]
        elif head_str == "omol":
            return self._head_omol(latent)  # type: ignore[no-any-return]
        return self._head_cp2k(latent)  # type: ignore[no-any-return]
