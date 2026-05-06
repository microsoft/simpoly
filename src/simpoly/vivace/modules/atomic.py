import torch.nn

from simpoly.vivace import constant


class AtomicEnergies(torch.nn.Embedding):
    def __init__(self, values: torch.Tensor, trainable: bool = False) -> None:
        super().__init__(
            num_embeddings=constant.MAX_ATOMIC_NUMBER + 1,
            embedding_dim=1,
            dtype=torch.get_default_dtype(),
        )
        assert values.shape == (constant.MAX_ATOMIC_NUMBER + 1, 1)
        self.weight.data = values.clone().detach().to(torch.get_default_dtype())
        self.weight.requires_grad = trainable

    def __repr__(self) -> str:
        """Print non-zero values of atomic energies embedding layer."""
        return (
            f"{self.__class__.__name__}("
            f"out_dim={self.embedding_dim}, "
            f"nonzero_energies={self._format_nonzero_values()}, "
            f"trainable={self.weight.requires_grad})"
        )

    def _format_nonzero_values(self) -> str:
        assert self.weight.shape == (constant.MAX_ATOMIC_NUMBER + 1, 1)
        e_zero = torch.zeros((1,), device=self.weight.device, dtype=self.weight.dtype)
        z_e_dict = {z: e.item() for z, e in enumerate(self.weight) if not torch.isclose(e, e_zero)}
        return "{" + ",".join(f"{z}:{e:.2f}" for z, e in z_e_dict.items()) + "}"


class AtomicEmbedding(torch.nn.Embedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(
            num_embeddings=constant.MAX_ATOMIC_NUMBER + 1,
            embedding_dim=embedding_dim,
            dtype=torch.get_default_dtype(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(out_dim={self.embedding_dim})"
