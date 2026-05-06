import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs;
    ICLR 2020. Equation (7)
    * new feature: log_scale_init, if True, the weights are initialized in logspace
    """

    def __init__(
        self,
        r_max: float,
        n_basis: int = 8,
        trainable: bool = False,
        log_scale_init: bool = False,
    ) -> None:
        super().__init__()

        if log_scale_init:
            bessel_weights = (
                torch.logspace(start=math.log10(1.0), end=math.log10(n_basis), steps=n_basis)
                * math.pi
                / r_max
            )
        else:
            bessel_weights = (
                np.pi
                / r_max
                * torch.linspace(
                    start=1.0,
                    end=n_basis,
                    steps=n_basis,
                    dtype=torch.get_default_dtype(),
                )
            )
        bessel_weights.requires_grad = trainable

        if trainable:
            self.bessel_weights = torch.nn.Parameter(
                bessel_weights,
                requires_grad=trainable,
            )
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer("r_max", torch.tensor(r_max))
        self.register_buffer("pre_factor", torch.sqrt(torch.tensor(2.0 / r_max)))
        self.register_buffer("n_basis", torch.tensor(n_basis))

    def forward(
        self,
        x: torch.Tensor,  # [..., 1]
    ) -> torch.Tensor:  # [..., num_basis]
        value: torch.Tensor = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        value = value / x
        value = value * self.pre_factor
        return value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"r_max={self.r_max.item():.3f}, "
            f"num_basis={self.n_basis.item()}, "
            f"trainable={self.bessel_weights.requires_grad}"
            ")"
        )


class GaussianBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        n_basis: int,
        r_min: float = 0.5,
        log_scale_init: bool = False,
        zero_init: bool = False,
        std_init: float = 0.2,
        diffusive_std: bool = False,
        eps: float = 1e-5,
        mean_trainable: bool = False,
        std_trainable: bool = False,
    ):
        super().__init__()

        assert not (
            log_scale_init and zero_init
        ), "log_scale_init and zero_init are mutually exclusive"

        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("r_min", torch.tensor(r_min, dtype=torch.get_default_dtype()))
        self.register_buffer("n_basis", torch.tensor(n_basis))

        means: torch.Tensor
        std_inv: torch.Tensor
        if log_scale_init:
            # first value is at 0.5, shortest natural bond is 0.7
            means = torch.logspace(np.log10(r_min), 0, n_basis, base=10)  # [0.5, 1]
            means = (means - r_min) / (1 - r_min) * (r_max - r_min)  # [0, r_max - 0.5]
            distances = means[1:] - means[:-1]
            # set std dev to half the distance to the next kernel
            std_inv = 1.0 / torch.tensor((distances / 2).tolist() + [distances[-1] / 2])
            means = means[None, :] + r_min  # [0.5, r_max]
            std_inv = std_inv[None, :]
        else:
            if zero_init:
                means = torch.zeros(1, n_basis)
                diffusive_std = True
            else:
                means = torch.linspace(r_min, r_max, n_basis)[None, :]
            if diffusive_std:
                std_inv = 5 * torch.linspace(1.0, 1.0 / n_basis, n_basis)[None, :]
            else:
                std_inv = 1.0 / (std_init * torch.ones(1, n_basis))

        means.requires_grad = mean_trainable

        if mean_trainable:
            self.means = torch.nn.Parameter(means)
        else:
            self.register_buffer("means", means)

        self.std_trainable = std_trainable
        if std_trainable:
            std_inv_sqrt = std_inv.sqrt()
            self.std_inv_sqrt = torch.nn.Parameter(std_inv_sqrt)  # ensures detach + grad
            self.register_buffer("std_inv", torch.tensor(float("nan")))  # should not be used!
        else:
            std_inv.requires_grad = False
            self.register_buffer("std_inv_sqrt", torch.tensor(float("nan")))  # should not be used
            self.register_buffer("std_inv", std_inv)

        self.register_buffer("prefactor", torch.tensor(1.0 / (2 * math.pi) ** 0.5))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std_trainable:
            # for numeric stability, we ensure that this quantity is always strictly positive
            std_inv = self.std_inv_sqrt**2 + self.eps
        else:
            std_inv = self.std_inv
        coeff = self.prefactor * std_inv
        rbf = torch.exp(-0.5 * ((x - self.means) * std_inv) ** 2)
        rbf = rbf * coeff
        return rbf


class SmoothBesselBasis(torch.nn.Module):
    def __init__(self, r_max: float, n_basis: int = 10):
        r"""Smooth Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        This is an orthogonal basis with first
        and second derivative at the cutoff
        equals to zero. The function was derived from the order 0 spherical Bessel
        function, and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: torch.Tensor distance tensor
            n_max: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with derivatives smooth at boundary

        Note: the code is copy from MatterSim. Performance not optimized
        """
        super().__init__()
        self.n_basis = n_basis
        n = torch.arange(0, n_basis).float()[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn_list = [torch.tensor(1.0).float()]
        for i in range(1, n_basis):
            dn_list.append(1 - en[0, i] / dn_list[-1])
        dn = torch.stack(dn_list)
        self.register_buffer("dn", dn)
        self.register_buffer("en", en)
        self.register_buffer("fnr_weights", fnr)
        self.register_buffer(
            "n_1_pi_cutoff",
            ((torch.arange(0, n_basis).float() + 1) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer(
            "n_2_pi_cutoff",
            ((torch.arange(0, n_basis).float() + 2) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x_1 = x * self.n_1_pi_cutoff
        x_2 = x * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.n_basis):
            gn.append(
                1
                / torch.sqrt(self.dn[i])
                * (fnr[:, i] + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1])
            )
        return torch.transpose(torch.stack(gn), 1, 0)


class ExpNormalSmearing(nn.Module):
    """Exponential normal smearing for radial basis functions.
    originally from PhysNet

    exp(-betas*(exp(alpha*(-x+r_min))-means)**2)
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 5.0,
        num_basis: int = 32,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.num_basis = num_basis
        self.trainable = trainable
        self.alpha = 5.0 / (r_max - r_min)
        means, neg_betas = self._initial_params()

        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("neg_betas", nn.Parameter(neg_betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("neg_betas", neg_betas)

    def _initial_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize means and neg_betas according to PhysNet defaults."""
        start_value = torch.exp(torch.scalar_tensor(-self.r_max + self.r_min))
        means = torch.linspace(start_value, 1, self.num_basis)
        means = means.unsqueeze(0)  # [1, num_basis]
        neg_betas = -torch.tensor([(2 / self.num_basis * (1 - start_value)) ** -2] * self.num_basis)
        neg_betas = neg_betas.unsqueeze(0)  # [1, num_basis]
        return means, neg_betas

    def reset_parameters(self) -> None:
        """Reset the trainable parameters to their initial values."""
        means, neg_betas = self._initial_params()
        self.means.data.copy_(means)
        self.neg_betas.data.copy_(neg_betas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.exp(
        #     self.neg_betas * (torch.exp(self.alpha * (-x + self.r_min)) - self.means) ** 2
        # )
        x = torch.add(input=torch.as_tensor(self.r_min * self.alpha), other=x, alpha=-self.alpha)
        x = torch.exp(x)
        x = x - self.means  # [n_edges, num_basis]
        x = x**2
        x = x * self.neg_betas
        x = torch.exp(x)
        return x


class MixExpNormSmoothBessel(nn.Module):
    """
    This mixes the Exponential Normal Smearing with the Smooth Bessel Basis.

    SmoothBesselBasis is used for the first 4 basis functions, and
    ExpNormalSmearing is used for the rest.

    The smooth bessel basis is especially useful for crystal structure. however, it is not suited for longer
    distances. The exponential normal smearing is used for longer distances.

    Therefore the r_max of the bessel basis is set to 5.0, and the r_max of the exp normal smearing is set to r_max
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 5.0,
        num_basis: int = 32,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        assert num_basis > 4, "num_basis must be greater than 4 for MixExpNormSmoothBessel"
        bessel_r_max = 5.0 if r_max > 5.0 else r_max
        self.bessel = SmoothBesselBasis(r_max=bessel_r_max, n_basis=4)
        self.exp_norm = ExpNormalSmearing(
            r_min=r_min,
            r_max=r_max,
            num_basis=num_basis - 4,
            trainable=trainable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp = self.exp_norm(x)
        bessel = self.bessel(x)
        basis = torch.cat([exp, bessel], dim=-1)
        return basis
