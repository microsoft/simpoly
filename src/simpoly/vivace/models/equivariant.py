import math
from typing import Final, Literal, Optional

import torch
from e3nn import o3
from torch.nn.modules.module import T

import simpoly.vivace.modules.ltp_cuequivariance as cue
from simpoly.vivace import keys
from simpoly.vivace.modules import (
    MakeWeightedChannels,
    ManualDotProduct,
    NoBiasMLP,
    QKVConcat,
    QKVExp,
    QKVSiLu,
    QKVSoftmax,
    UnweightedTPDense,
    UnweightedTPUnrollDense,
)
from simpoly.vivace.utils.scatter import scatter_sum

from .layers import is_aoti_export_mode as _aoti_export_mode  # noqa: F401 (Note:flag)
from .strided_linear import LinearWrapper  # type: ignore[attr-defined]

O3_FULL: Final[str] = "o3_full"
O3_RESTRICTED: Final[str] = "o3_restricted"
SO3: Final[str] = "so3"
PARITY_SETTING_TYPES = Literal["o3_full", "o3_restricted", "so3"]


class MockInnerProduct(torch.nn.Module):
    """
    A mock inner product that does nothing, used to make sure jit scripting works consistently
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


class VivaceTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        use_cuequivariance: bool,
    ) -> None:
        super().__init__()

        assert irreps_out[0].ir.is_scalar()
        irreps_tp_out, instr = self.create_instructions_from_irreps(
            irreps_in1, irreps_in2, irreps_out
        )
        self.mul = irreps_tp_out[0].mul
        self.n_irreps = len(irreps_tp_out)

        self.tp_kwargs = dict(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_tp_out,
            instructions=instr,
        )
        self.tp: torch.nn.Module
        self.linear_out: torch.nn.Module
        self.use_cuequivariance = use_cuequivariance
        if use_cuequivariance:
            self.tp = cue.LTPCuEquivarianceTwist(**self.tp_kwargs)
            self.linear_out = cue.LinearWrapperTwist(
                irreps_tp_out,
                irreps_out,
                dtype=torch.get_default_dtype(),
            )
        else:
            self.tp = UnweightedTPUnrollDense(**self.tp_kwargs)
            self.linear_out = LinearWrapper(
                irreps_tp_out,
                irreps_out,
                shared_weights=True,
                internal_weights=True,
            )

    def create_instructions_from_irreps(
        self, irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, irreps_out: o3.Irreps
    ) -> tuple[o3.Irreps, list[tuple[int, int, int]]]:
        instr = []
        irreps_tp_out = [(mul, ir) for mul, ir in irreps_out]
        for i_out, (_, ir_out) in enumerate(irreps_out):
            for i_in1, (_, ir_in1) in enumerate(irreps_in1):
                for i_in2, (_, ir_in2) in enumerate(irreps_in2):
                    if ir_out in ir_in1 * ir_in2:
                        instr.append((i_in1, i_in2, i_out))
        irreps_tp_out = o3.Irreps(irreps_tp_out)
        return irreps_tp_out, instr

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # if self.use_cuequivariance: input should be
        tp_out = self.tp(x, y)
        feats = self.linear_out(tp_out)
        return feats  # type: ignore[no-any-return]

    def train(self: T, mode: bool = True) -> T:
        self = super().train(mode)
        tp: torch.nn.Module
        if not self.use_cuequivariance:
            if mode:
                device = next(self.tp.buffers()).device
                tp = UnweightedTPUnrollDense(
                    irreps_in1=self.tp_kwargs["irreps_in1"],
                    irreps_in2=self.tp_kwargs["irreps_in2"],
                    irreps_out=self.tp_kwargs["irreps_out"],
                    instructions=self.tp_kwargs["instructions"],
                )
                self.tp.to(device)
            else:
                device = next(self.tp.buffers()).device
                tp = UnweightedTPDense(
                    irreps_in1=self.tp_kwargs["irreps_in1"],
                    irreps_in2=self.tp_kwargs["irreps_in2"],
                    irreps_out=self.tp_kwargs["irreps_out"],
                    instructions=self.tp_kwargs["instructions"],
                )
                self.tp = tp
                self.tp.to(device)
        return self


class InvariantCrossAttention2EquivariantNode(torch.nn.Module):
    """
    Performs an invariant-to-equivariant cross-attention operation.

    This module takes invariant node and edge features and produces equivariant node features.
    The attention mechanism uses node features to form queries, and edge features to form
    keys and values. The resulting attention weights are used to weight the spherical harmonics
    of the edges, which are then scattered to the nodes to produce equivariant features.
    The module can also optionally update the invariant edge features based on the new
    equivariant node features.

    Args:
        irreps_in (o3.Irreps): Input equivariant irreps for the nodes.
        irreps_out (o3.Irreps): Output equivariant irreps for the nodes.
        irreps_edge_sh (o3.Irreps): Irreps of the spherical harmonics for the edges.
        n_invariant_features (int): Dimensionality of the latent invariant features.
        n_equivariant_features (int): Multiplicity for the environment-weighted features.
        n_attn_heads (int): Number of attention heads.
        attn_head_dim (int, optional): Dimensionality of each attention head. If None,
            it is calculated as `n_invariant_features // n_attn_heads`. Defaults to None.
        use_l0_only (bool): If True, only use the l=0 component of the equivariant features.
            Defaults to True.
        use_cuequivariance (bool): If True, use the CUDA-accelerated version of the
            tensor products. Defaults to False.
        update_edge (bool): If True, update the invariant edge features based on the
            new equivariant node features. Defaults to True.
        attention_type (str): The transformation to apply to the queries, keys, and values.
            One of "exp", "concat", "silu", "softmax". Defaults to "exp".
        n_mlp_layer (int): Number of layers in the post-attention MLP. Defaults to 1.

    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_sh: o3.Irreps = o3.Irreps("1x0e+1x1o+1x2e"),
        n_invariant_features: int = 128,
        n_equivariant_features: int = 32,
        n_attn_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        use_l0_only: bool = True,
        use_cuequivariance: bool = False,
        update_edge: bool = True,
        attention_type: str = "exp",
        n_mlp_layer: int = 1,
    ) -> None:

        super().__init__()

        ## go through irreps_in and irreps_edge_sh to find common irreps and their index
        # get index_in and index_edge_sh, so one can do a dot product
        # between the two

        self.n_attn_heads = n_attn_heads
        if attn_head_dim is None:
            if n_invariant_features % n_attn_heads == 0:
                attn_head_dim = n_invariant_features // n_attn_heads
            else:
                attn_head_dim = max(8, int(n_invariant_features / n_attn_heads) * 2)
        self.attn_head_dim = int(attn_head_dim)
        self.scaling_factor = 1 / math.sqrt(self.attn_head_dim)

        linear_kwargs = dict(
            in_features=n_invariant_features,
            out_features=self.n_attn_heads * self.attn_head_dim,
        )
        self.query_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # node_i -> Q_i
        self.key_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # edge_ij -> K_ij
        self.value_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # edge_ij -> V_ij

        irreps_attr = o3.Irreps([(n_equivariant_features, ir) for _, ir in irreps_edge_sh])
        self.n_env_weight = irreps_attr.num_irreps
        # this bias has to be false because it involved edge quantities
        # and envelope is multiplied before it
        self.out_lin = torch.nn.Linear(
            self.n_attn_heads * self.attn_head_dim,
            self.n_env_weight,
            bias=False,
        )

        self.use_cuequivariance = use_cuequivariance
        self.env_weighted: torch.nn.Module
        if self.use_cuequivariance:
            self.env_weighted = cue.MakeWeightedChannelsTwist(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )
        else:
            self.env_weighted = MakeWeightedChannels(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )
        self.tp = VivaceTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_attr,
            irreps_out=irreps_out,
            use_cuequivariance=use_cuequivariance,
        )
        self.equiv_feature_resnet = True if irreps_in == irreps_out else False

        # place holder when no edge update is needed, need to be a callable with three arguments
        # so that it can be used in the same way as inner_product
        self.update_edge = False
        self.inner_product: torch.nn.Module = MockInnerProduct()
        if update_edge:
            if len(irreps_out) > 1:
                self.update_edge = True
                if use_cuequivariance:
                    self.inner_product = cue.DotProductCuEquivarianceTwist(
                        irreps_in1=irreps_out,
                        irreps_in2=irreps_edge_sh,
                    )
                else:
                    self.inner_product = ManualDotProduct(
                        irreps_in=irreps_out,
                        irreps_edge_sh=irreps_edge_sh,
                    )
            else:
                self.update_edge = False
                # place holder when no edge update is needed, need to be a callable with three arguments
                # so that it can be used in the same way as inner_product
                self.inner_product = MockInnerProduct()

        self.n_irreps_attr = irreps_attr.num_irreps
        self.n_irreps_out = irreps_out.num_irreps

        self.attention_type: torch.nn.Module
        transform_types = {
            "softmax": QKVSoftmax,
            "exp": QKVExp,
            "silu": QKVSiLu,
            "concat": QKVConcat,
        }

        # eps is default to 1.0 because softmax rely on this one
        # to not explode when all entries vanishes to zeros
        self.attention_type = transform_types[attention_type](
            num_attn_heads=n_attn_heads,
            attn_head_dim=attn_head_dim,
            eps=1.0,
        )
        self.out_lin2 = torch.nn.Linear(
            n_equivariant_features,  # irreps_out.n_irreps,
            n_invariant_features,
            bias=False,
        )

        hidden_dim = 256 if n_invariant_features < 256 else n_invariant_features
        self.post_attn_mlp = NoBiasMLP(
            # input_dim=n_equivariant_features, # irreps_out.n_irreps,
            input_dim=n_invariant_features,
            output_dim=n_invariant_features,
            hidden_dims=[hidden_dim] * n_mlp_layer,
        )
        self.edge_filter_mlp = NoBiasMLP(
            input_dim=n_invariant_features,
            output_dim=n_invariant_features,
            hidden_dims=[n_invariant_features] * n_mlp_layer,
        )

        self.use_l0_only = use_l0_only

        self.layer_norm = torch.nn.LayerNorm(
            n_invariant_features,
            elementwise_affine=False,
            eps=1e-5,
        )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        envelope = data[keys.EDGE_LENGTH_ENVELOPE_EQUI]
        edge_mask = data[keys.EDGE_MASK]
        receiver = data[keys.RECEIVER]
        edge_length_embed = data[keys.EDGE_EMBEDDING_EQUIV_RMAX]

        spherical_harmonics = data[keys.EDGE_ATTRS]
        full_edge_invariant = data[keys.EDGE_INVARIANT]

        node_invariant = data[keys.NODE_INVARIANT]
        node_equivariant = data[keys.NODE_EQUIVARIANT]

        # NOTE: 1 means "not masked", i.e. "keep"
        # Note: the original branch ladder
        # if mask.sum() == 0: return data
        # elif mask.sum() == mask.numel(): has_masking = False
        # else: has_masking = True
        # is two data-dependent guards on `mask.sum()` that block torch.export.
        # Under AOTI-export we collapse to the *masked* path (always index by
        # `edge_mask`) and use the SymInt that's already in scope (the leading
        # dim of the `equiv_envelope` tensor produced by InitHeader, which
        # equals `mask.sum()` by construction) so the exporter sees a clean
        # shape symbol rather than a 0-d FakeTensor from `mask.sum()`.
        # The `> 1` bound (rather than `> 0`) avoids cueq's
        # `_handle_batch_dim_auto` data-dependent `Eq(u1, 1)` broadcast guard.
        # The mliap.py OUTER fallback enforces these invariants at runtime;
        # the `mask.sum() == numel` shortcut is dead in the deployed cueq-0.8
        # model (r_max=6.5, equiv_r_max=3.8 ⇒ always some edges masked).
        if _aoti_export_mode():
            n_equiv_edges = envelope.shape[0]
            torch._check(n_equiv_edges > 1)
            has_masking = True
        else:
            mask_sum = edge_mask.sum()
            if mask_sum == 0:
                return data
            elif mask_sum == edge_mask.numel():
                has_masking = False
            else:
                has_masking = True

        edge_invariant = full_edge_invariant
        if has_masking:
            receiver = data[keys.RECEIVER][edge_mask]
            edge_invariant = edge_invariant[edge_mask]
            if _aoti_export_mode():
                # The boolean-mask indexing creates a *fresh* unbacked SymInt
                # for the new leading dim; constrain it the same way as
                # `n_equiv_edges` so cueq's `_handle_batch_dim_auto` can resolve
                # `Eq(u, 1)` further down the call chain.
                torch._check(edge_invariant.shape[0] > 1)
            # `spherical_harmonics` does not need to be masked because
            # it is already masked in the data
            # same to `envelope`

        query_i = self.query_lin(node_invariant)
        query_i = query_i.index_select(0, receiver)
        key_ij = self.key_lin(edge_invariant)
        value_ij = self.value_lin(edge_invariant)

        # note, because here we need to scatter sum to node feature, envelope is required
        env_weight = self.attention_type(
            receiver,
            query_i,
            key_ij,
            value_ij,
            envelope,
            dim_size=node_invariant.size(0),
        )
        env_weight = self.out_lin(env_weight)

        tmp_edge_equivariant = self.env_weighted(spherical_harmonics, env_weight)

        tmp_node_equivariant = scatter_sum(
            src=tmp_edge_equivariant,
            index=receiver,
            dim=0,
            dim_size=node_invariant.size(0),
        )

        # data["tmp_node_equivariant"] = tmp_node_equivariant
        new_node_equivariant = self.tp(
            node_equivariant,
            tmp_node_equivariant,
        )  # [node, channel, irreps( 0-1 for 0e, 1-4 for 1o ...]
        # [node, irreps, channel] if cuequivariance

        # normalization with mean over channel with irreps forbenius norm
        normalizing_factor = torch.square(new_node_equivariant)  # [node, channel, irreps]
        # sum over irreps so it is invariant
        if self.use_cuequivariance:
            normalizing_factor = torch.sum(normalizing_factor, dim=-2)  # [node, channel]
        else:
            normalizing_factor = torch.sum(normalizing_factor, dim=-1)  # [node, channel]
        normalizing_factor = torch.mean(normalizing_factor, dim=-1)  # [node,]
        normalizing_factor = normalizing_factor + 1e-5  # [node,]
        normalizing_factor = normalizing_factor.unsqueeze(-1).unsqueeze(-1)  # [node, 1, 1]

        new_node_equivariant = new_node_equivariant / normalizing_factor  # [node, channel, irreps]

        # when the node equivariant contains more than 0e irreps
        if self.update_edge:
            tmp_edge = new_node_equivariant[receiver]
            new_edge_invariant = self.inner_product(
                tmp_edge,  # [ne, nf, nc]
                spherical_harmonics,  # [ne, nf, nc]
            )
            new_edge_invariant = self.out_lin2(new_edge_invariant)

            # note, edge feature never multiply with envelope before final MLP reading
            edge_filter = self.edge_filter_mlp(edge_length_embed)

            # new_edge_invariant = new_edge_invariant * edge_filter
            # new_edge_invariant = edge_invariant + new_edge_invariant
            new_edge_invariant = torch.addcmul(
                edge_invariant,  # [ne, nc]
                edge_filter,  # [ne, nc]
                new_edge_invariant,  # [ne, nc]
            )
            new_edge_invariant = self.post_attn_mlp(new_edge_invariant)
            # note, we need to multiply by envelope because equivariant features may have a different cutoff value
            # only setting invariant MLP to no bias is not sufficient
            # new_edge_invariant = new_edge_invariant * envelope
            # new_edge_invariant = edge_invariant + new_edge_invariant
            new_edge_invariant = torch.addcmul(
                edge_invariant,  # [ne, nc]
                envelope,  # [ne, nc]
                new_edge_invariant,  # [ne, nc]
            )

            if has_masking:
                temp_edge_invariant = full_edge_invariant.clone()
                temp_edge_invariant[edge_mask] = new_edge_invariant
                data[keys.EDGE_INVARIANT] = temp_edge_invariant
            else:
                data[keys.EDGE_INVARIANT] = new_edge_invariant

        # for the last layer, when output is 0e, we only update the node invariant
        else:
            if new_node_equivariant.dim() == 3:
                if self.use_cuequivariance:
                    new_node_invariant = new_node_equivariant[:, 0, :]
                else:
                    new_node_invariant = new_node_equivariant[:, :, 0]
            else:
                new_node_invariant = new_node_equivariant
            new_node_invariant = self.out_lin2(new_node_invariant)
            new_node_invariant = self.layer_norm(new_node_invariant)
            new_node_invariant = self.post_attn_mlp(new_node_invariant)
            data[keys.NODE_INVARIANT] = new_node_invariant + node_invariant

        if self.equiv_feature_resnet:
            data[keys.NODE_EQUIVARIANT] = node_equivariant + new_node_equivariant
        else:
            data[keys.NODE_EQUIVARIANT] = new_node_equivariant

        return data


class InvariantCrossAttention2EquivariantEdge(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_sh: o3.Irreps = o3.Irreps("1x0e+1x1o+1x2e"),
        n_invariant_features: int = 128,
        n_equivariant_features: int = 32,
        n_attn_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        use_l0_only: bool = True,
        use_cuequivariance: bool = False,
        attention_type: str = "exp",
        n_mlp_layer: int = 1,
    ) -> None:
        super().__init__()

        self.n_attn_heads = n_attn_heads
        if attn_head_dim is None:
            if n_invariant_features % n_attn_heads == 0:
                attn_head_dim = n_invariant_features // n_attn_heads
            else:
                attn_head_dim = max(8, int(n_invariant_features / n_attn_heads) * 2)
        self.attn_head_dim = attn_head_dim
        self.scaling_factor = 1 / math.sqrt(self.attn_head_dim)

        linear_kwargs = dict(
            in_features=n_invariant_features,
            out_features=self.n_attn_heads * self.attn_head_dim,
        )
        self.node_query_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # node_i -> Q_i
        self.edge_key_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # edge_ij -> K_ij
        self.edge_value_lin = torch.nn.Linear(**linear_kwargs, bias=False)  # edge_ij -> V_ij

        irreps_attr = o3.Irreps([(n_equivariant_features, ir) for _, ir in irreps_edge_sh])
        self.n_env_weight = irreps_attr.num_irreps
        # has to be no bias because envelope will set weights to zero when close to cutoff
        self.out_lin = torch.nn.Linear(
            self.n_attn_heads * self.attn_head_dim,
            self.n_env_weight,
            bias=False,
        )

        self.use_cuequivariance = use_cuequivariance
        self.env_weighted: torch.nn.Module
        if self.use_cuequivariance:
            self.env_weighted = cue.MakeWeightedChannelsTwist(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )
        else:
            self.env_weighted = MakeWeightedChannels(
                irreps_in=irreps_edge_sh,
                multiplicity_out=n_equivariant_features,
            )
        self.tp = VivaceTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_attr,
            irreps_out=irreps_out,
            use_cuequivariance=use_cuequivariance,
        )
        self.edge_features_resnet = True if irreps_in == irreps_out else False

        self.n_irreps_attr = irreps_attr.num_irreps
        self.n_irreps_out = irreps_out.num_irreps

        self.attention_type: torch.nn.Module
        transform_types = {
            "softmax": QKVSoftmax,
            "exp": QKVExp,
            "silu": QKVSiLu,
            "concat": QKVConcat,
        }
        self.attention_type = transform_types[attention_type](
            num_attn_heads=n_attn_heads,
            attn_head_dim=attn_head_dim,
            eps=0.001,
        )

        # since this is acting on edge equivariant feature
        # we do not have a regulation to make sure it goes zero
        # will have to use NoBias
        hidden_dim = 256 if n_invariant_features < 256 else n_invariant_features
        self.post_attn_mlp = NoBiasMLP(
            input_dim=n_equivariant_features,  # irreps_out.num_irreps,
            output_dim=n_invariant_features,
            hidden_dims=[hidden_dim] * n_mlp_layer,
        )

        self.use_l0_only = use_l0_only

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        envelope = data[keys.EDGE_LENGTH_ENVELOPE_EQUI]
        edge_mask = data[keys.EDGE_MASK]
        receiver = data[keys.RECEIVER]

        spherical_harmonics = data[keys.EDGE_ATTRS]
        full_edge_invariant = data[keys.EDGE_INVARIANT]

        node_invariant = data[keys.NODE_INVARIANT]
        edge_equivariant = data[keys.EDGE_EQUIVARIANT]
        # NOTE: 1 means "not masked", i.e. "keep"
        # Note: same surgery as `EquivariantSelfAttention.forward`
        # above. Collapse the branch ladder to the masked always-take path
        # under AOTI-export and use `envelope.shape[0]` (the existing SymInt)
        # rather than `edge_mask.sum()` (a 0-d FakeTensor under tracing).
        if _aoti_export_mode():
            n_equiv_edges = envelope.shape[0]
            torch._check(n_equiv_edges > 1)
            has_masking = True
        else:
            mask_sum = edge_mask.sum()
            if mask_sum == 0:
                return data
            elif mask_sum == edge_mask.numel():
                has_masking = False
            else:
                has_masking = True

        edge_invariant = full_edge_invariant
        if has_masking:
            receiver = data[keys.RECEIVER][edge_mask]
            edge_invariant = edge_invariant[edge_mask]
            if _aoti_export_mode():
                torch._check(edge_invariant.shape[0] > 1)

        query_i = self.node_query_lin(node_invariant)
        query_i = query_i.index_select(0, receiver)
        key_ij = self.edge_key_lin(edge_invariant)
        value_ij = self.edge_value_lin(edge_invariant)

        # we need envelope here to make sure the attention is zeroed out
        # so when it is used in the later scatter sum, the close to cutoff contribution is zero
        env_weight = self.attention_type(
            receiver,
            query_i,
            key_ij,
            value_ij,
            envelope,
            dim_size=node_invariant.size(0),
        )
        env_weight = self.out_lin(env_weight)

        tmp_edge_equivariant = self.env_weighted(spherical_harmonics, env_weight)
        tmp_node_equivariant = scatter_sum(
            src=tmp_edge_equivariant,
            index=receiver,
            dim=0,
            dim_size=node_invariant.size(0),
        )

        expanded_env = tmp_node_equivariant[receiver]

        new_edge_equivariant = self.tp(edge_equivariant, expanded_env)

        if self.edge_features_resnet:
            new_edge_equivariant = edge_equivariant + new_edge_equivariant

        if new_edge_equivariant.dim() == 3:
            if self.use_l0_only:
                if self.use_cuequivariance:
                    new_edge_invariant = new_edge_equivariant[:, 0, :]
                else:
                    new_edge_invariant = new_edge_equivariant[:, :, 0]
            else:
                if self.use_cuequivariance:
                    new_edge_invariant = torch.sum(torch.square(new_edge_equivariant), dim=-2)
                else:
                    new_edge_invariant = torch.sum(torch.square(new_edge_equivariant), dim=-1)
        else:
            new_edge_invariant = new_edge_equivariant
        new_edge_invariant = self.post_attn_mlp(new_edge_invariant)
        # new_edge_invariant = new_edge_invariant * envelope
        # new_edge_invariant = new_edge_invariant + edge_invariant
        new_edge_invariant = torch.addcmul(edge_invariant, envelope, new_edge_invariant)

        data[keys.EDGE_EQUIVARIANT] = new_edge_equivariant
        if has_masking:
            # has masking means the equivariant cutoff is different
            # we need to make sure its contribution is zero when it is closer to the smaller cutoff
            temp_edge_invariant = full_edge_invariant.clone()
            temp_edge_invariant[edge_mask] = new_edge_invariant
            data[keys.EDGE_INVARIANT] = temp_edge_invariant
        else:
            data[keys.EDGE_INVARIANT] = new_edge_invariant
        return data
