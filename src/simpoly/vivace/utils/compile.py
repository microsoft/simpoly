from types import TracebackType
from typing import Optional, TypeAlias

import torch

TensorDict: TypeAlias = dict[str, torch.Tensor]
OptionalTensorDict: TypeAlias = dict[str, Optional[torch.Tensor]]


def tensor_dict_allclose(
    a: OptionalTensorDict,
    b: OptionalTensorDict,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    a_keys = set(a.keys())
    b_keys = set(b.keys())

    if a_keys != b_keys:
        return False

    for key in a_keys:
        v_a, v_b = a[key], b[key]
        if v_a is None and v_b is None:
            pass
        elif v_a is None or v_b is None:
            raise ValueError(f"Key '{key}' is None in one of the dicts but not the other")
        else:
            if not torch.allclose(v_a, v_b, rtol=rtol, atol=atol, equal_nan=equal_nan):
                return False
    return True


class TrainingModeManager:
    # Switching to eval mode has an effect only on certain modules. See documentations
    # of particular modules for details of their behaviors in training/evaluation mode,
    # if they are affected, e.g. Dropout, BatchNorm, etc.
    def __init__(self, model: torch.nn.Module, train_mode: bool) -> None:
        self.model = model
        self.train_mode = train_mode
        self.original_mode = model.training

    def __enter__(self) -> None:
        self.model.train(self.train_mode)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.model.train(self.original_mode)
