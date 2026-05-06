import math

import torch

from simpoly.vivace import modules


class QKVTransform(torch.nn.Module):
    def __init__(
        self,
        num_attn_heads: int,
        attn_head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_attn_heads = num_attn_heads
        self.attn_head_dim = attn_head_dim
        self.scaling_factor = 1 / math.sqrt(self.attn_head_dim)
        self.n_latent = self.attn_head_dim * num_attn_heads
        self.eps = eps


class QKVSoftmax(QKVTransform):
    """
    update edge feature e_ij based on query, key and value ij
    equation:
    e_ij' = {softmax_i[Q_ij @ K_ij / sqrt(d)] @ V_ij}

    The softmax is taken over the first dimension (i) of the tensor

    NOTE: the equation does not write out the multi-head part
    """

    def forward(
        self,
        receiver: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cutoff: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:

        query = query.view(query.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        key = key.view(key.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        scaled_dot = torch.sum(query * key, dim=-1) * self.scaling_factor
        value = value.view(value.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))

        # Note: thread dim_size to softmax_cutoff to avoid an `index.max()`
        # CPU↔GPU sync (CUDAGraph capture prerequisite).
        attn_ij: torch.Tensor = modules.softmax_cutoff(
            scaled_dot, cutoff, receiver, dim=0, dim_size=dim_size, eps=self.eps
        )
        value_prime_ij = attn_ij.unsqueeze(-1) * value
        value_prime_ij = value_prime_ij.view(value_prime_ij.shape[:-2] + (self.n_latent,))
        return value_prime_ij


class QKVExp(QKVTransform):
    """
    update edge feature e_ij based on node feature n_i and e_ij
    equation:
    e_ij' = {exp[-0.5|Q_ij - K_ij|_d^2 / sqrt(d)] @ V_ij} * cutoff_ij

    where the norm is taken over the last dimension (d) of the tensor

    NOTE: the equation does not write out the multi-head part
    """

    def forward(
        self,
        receiver: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cutoff: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        # Note: dim_size accepted for signature parity with QKVSoftmax;
        # unused here because QKVExp is dense (no per-receiver normalization).
        del dim_size
        query = query.view(query.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        key = key.view(key.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        value = value.view(value.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        diff = query - key
        diff_squared_sum = diff.pow(2).sum(dim=-1)
        kernel = torch.exp(-0.5 * diff_squared_sum * self.scaling_factor)
        value_prime = kernel.unsqueeze(-1) * value
        value_prime = value_prime.view(value_prime.shape[:-2] + (self.n_latent,))
        return value_prime * cutoff


class QKVSiLu(QKVTransform):
    """
    update edge feature e_ij based on node feature n_i and e_ij
    equation:
    e_ij' = SiLu[Q_i @ K_ij / sqrt(d)] @ V_ij * cutoff_ij

    where the inner product is taken over the last dimension (d) of the tensor

    This attention was used in TorchMD

    NOTE: the equation does not write out the multi-head part
    """

    def forward(
        self,
        receiver: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cutoff: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        # Note: dim_size accepted for signature parity; unused here.
        del dim_size
        query = query.view(query.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        key = key.view(key.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        kernel = torch.nn.functional.silu(torch.sum(query * key, dim=-1) * self.scaling_factor)
        value = value.view(value.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        value_prime = kernel.unsqueeze(-1) * value
        value_prime = value_prime.view(value_prime.shape[:-2] + (self.n_latent,))
        return value_prime * cutoff


class QKVConcat(QKVTransform):
    """
    update edge feature e_ij based on
    equation:
    e_ij' = SiLu[Lin(Q_ij // K_ij) @ V_ij] * cutoff_ij
    """

    def __init__(
        self,
        num_attn_heads: int,
        attn_head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            num_attn_heads=num_attn_heads,
            attn_head_dim=attn_head_dim,
            eps=eps,
        )

        self.linear = torch.nn.Linear(
            in_features=self.attn_head_dim * 2, out_features=self.attn_head_dim, bias=False
        )

    def forward(
        self,
        receiver: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cutoff: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        # Note: dim_size accepted for signature parity; unused here.
        del dim_size

        query = query.view(query.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        key = key.view(key.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        kernel = torch.cat([query, key], dim=-1)
        kernel = self.linear(kernel)
        kernel = torch.nn.functional.silu(kernel)
        value = value.view(value.shape[:-1] + (self.num_attn_heads, self.attn_head_dim))
        value_prime_ij = kernel * value
        value_prime_ij = value_prime_ij.view(value_prime_ij.shape[:-2] + (self.n_latent,))
        return value_prime_ij * cutoff
