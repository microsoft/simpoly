import math

import torch


def shifted_soft_plus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)


class ShiftedSoftPlus(torch.nn.Module):
    def __init__(self) -> None:
        super(ShiftedSoftPlus, self).__init__()
        self.shift = math.log(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift
