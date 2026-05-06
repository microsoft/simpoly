import contextlib
import logging
import typing as ty

import numpy as np
import numpy.typing as npt
import torch

LOG = logging.getLogger(__name__)


def convert_to_dtype(s: str | torch.dtype) -> torch.dtype:
    name_to_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    if isinstance(s, torch.dtype):
        return s

    return name_to_dtype[s]


def to_numpy(t: torch.Tensor) -> npt.NDArray[ty.Any]:
    np_t: npt.NDArray[ty.Any] = t.detach().cpu().numpy()
    return np_t


def count_parameters(module: torch.nn.Module, only_trainable: bool = False) -> int:
    return int(
        sum(
            np.prod(p.shape) for p in module.parameters() if (p.requires_grad or not only_trainable)
        )
    )


def get_default_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device_type(device_type: str | None) -> torch.device:
    if device_type is None:
        device_type = get_default_device_type()

    assert device_type in {"cpu", "cuda"}
    return torch.device(device_type)


def tensor_dict_to_device(
    d: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    new = {}

    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new[k] = v.to(device)
        elif isinstance(v, dict):
            new[k] = tensor_dict_to_device(v, device)
        else:
            new[k] = v

    return new


def tensor_dict_to_cpu(d: dict[str, ty.Any]) -> dict[ty.Any, ty.Any]:
    return tensor_dict_to_device(d, torch.device("cpu"))


def get_dtype(model: torch.nn.Module) -> torch.dtype:
    # We assume all parameters are of the same dtype
    return next(model.parameters()).dtype


@contextlib.contextmanager
def torch_default_dtype_context(dtype: torch.dtype) -> ty.Iterator[None]:
    """Set `torch.get_default_dtype()` for the duration of a with block, cleaning up with
    a `finally`.
    Note: this is NOT thread safe, since `torch.set_default_dtype()` is not thread safe.
    """
    orig_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(orig_default_dtype)


@contextlib.contextmanager
def freeze_context(model: torch.nn.Module) -> ty.Generator[None, None, None]:
    """Context manager to freeze the model parameters and restore them after yielding."""
    params_grad_dict = {}
    for param in model.parameters():
        params_grad_dict[param] = param.requires_grad
        param.requires_grad = False
    yield
    for param, requires_grad in params_grad_dict.items():
        param.requires_grad = requires_grad


def convert_data_dtype(
    data: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    new_data: dict[str, torch.Tensor] = {}
    for k, v in data.items():
        if v.is_floating_point():
            v = v.to(dtype=dtype)
        new_data[k] = v
    return new_data
