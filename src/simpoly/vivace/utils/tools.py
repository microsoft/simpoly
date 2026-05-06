import logging
import os
import random
import string
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch

PathLike = Union[str, os.PathLike[str]]
LOG = logging.getLogger(__name__)


def get_utc_now() -> str:
    # Timestamp including milliseconds
    return datetime.now(timezone.utc).strftime(r"%Y%m%dT%H%M%S.%f")[:-3] + "Z"


def ensure_dir_exists(path: PathLike) -> str:
    """
    Resolves the full path, and ensure that a directory is located there.
    Creating the directory if necessary.
    Raises exception if the path or any parent is pointing at a file.
    """
    path = os.path.realpath(path)
    if os.path.isfile(path):
        raise ValueError("pointing to a file")
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def generate_random_dir_name(seed: Optional[int] = None, length: int = 10) -> str:
    rng = random.Random(seed)
    random_dir_name = "".join(
        rng.choice(string.ascii_letters + string.digits) for _ in range(length)
    )
    return random_dir_name


def check_allclose_detailed(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    name: str,
    rtol: float = 1e-04,
    atol: float = 1e-08,
    force_print: bool = False,
) -> bool:
    """
    Check allclose and provide detailed failure information.

    Parameters:
        a (npt.ArrayLike): First array to compare.
        b (npt.ArrayLike): Second array to compare.
        name (str): Name of the comparison for logging purposes.
        rtol (float): Relative tolerance. Default is 1e-04.
        atol (float): Absolute tolerance. Default is 1e-08.
        force_print (bool): If True, forces detailed failure information to be printed
            regardless of the comparison result. Default is False.

    Returns:
        bool: True if all elements are within tolerance, False otherwise.
    """
    # Convert inputs to numpy arrays for consistent handling
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    abs_diff = np.abs(a_arr - b_arr)
    tolerance_threshold = atol + rtol * np.abs(b_arr)

    # Check which elements pass/fail
    within_tolerance = abs_diff <= tolerance_threshold
    n_total = abs_diff.size
    n_pass = np.sum(within_tolerance)
    n_fail = n_total - n_pass

    # Find the worst violator
    violation_ratio = abs_diff / tolerance_threshold
    max_violation = np.max(violation_ratio)
    max_violation_idx = np.unravel_index(np.argmax(violation_ratio), violation_ratio.shape)

    print(f"\n=== {name} allclose analysis ===")
    print(f"Shape: {a_arr.shape}")
    print(f"Elements: {n_pass}/{n_total} pass, {n_fail} fail")

    if (n_fail > 0) or force_print:
        # Show details of worst violation
        flat_idx = np.ravel_multi_index(max_violation_idx, violation_ratio.shape)

        # Get values at worst violation point
        tolerance_at_worst = tolerance_threshold.flat[flat_idx]
        diff_at_worst = abs_diff.flat[flat_idx]
        ref_val_at_worst = np.abs(b_arr).flat[flat_idx]
        a_val_at_worst = a_arr.flat[flat_idx]
        b_val_at_worst = b_arr.flat[flat_idx]

        print(f"Worst violation at index {max_violation_idx}:")

        # Handle different data types based on shape
        if a_arr.ndim == 0 or (a_arr.ndim == 1 and a_arr.size == 1):
            # Scalar values (energy)
            print(f"  {name} value comparison:")
            print(f"  B (LAMMPS): {b_val_at_worst:.6e}")
            print(f"  A (Python): {a_val_at_worst:.6e}")

        elif a_arr.shape == (3, 3):
            # 3x3 tensor (pressure, virial, etc.)
            tensor_components = [["xx", "xy", "xz"], ["yx", "yy", "yz"], ["zx", "zy", "zz"]]
            i, j = max_violation_idx
            component_name = tensor_components[i][j]
            print(f"  {name} tensor {component_name} component:")
            print(f"  B (LAMMPS) {name.lower()} tensor:")
            print(f"    {b_arr}")
            print(f"  A (Python) {name.lower()} tensor:")
            print(f"    {a_arr}")
            print(f"  Component [{i},{j}] ({component_name}):")
            print(f"    B value: {b_val_at_worst:.6e}")
            print(f"    A value: {a_val_at_worst:.6e}")

        elif a_arr.ndim == 2 and a_arr.shape[1] == 3:
            # Forces: (n_atoms, 3)
            atom_idx = max_violation_idx[0]
            coord_names = ["x", "y", "z"]
            coord_idx = max_violation_idx[1]
            print(f"  {name} for atom {atom_idx}, {coord_names[coord_idx]} coordinate:")
            print(f"  B vector: {b_arr[atom_idx]}")
            print(f"  A vector: {a_arr[atom_idx]}")

        else:
            # Generic multi-dimensional array
            print(f"  {name} comparison at index {max_violation_idx}:")
            print(f"  B value: {b_val_at_worst:.6e}")
            print(f"  A value: {a_val_at_worst:.6e}")

        print(f"  Actual absolute difference: {diff_at_worst:.6e}")
        print(f"  Actual relative difference: {diff_at_worst / ref_val_at_worst:.6e}")
        print(f"  Required tolerance: {tolerance_at_worst:.6e}")
        print(f"  Violation factor: {max_violation:.2f}x")
        print(f"  Reference value: {ref_val_at_worst:.6e}")
        print(f"  rtol={rtol:.2e}, atol={atol:.2e}")

    is_close = np.allclose(a_arr, b_arr, rtol=rtol, atol=atol)
    print(f"Overall allclose result: {'PASS' if is_close else 'FAIL'}")
    return is_close


def assert_outputs_equal(
    output_a: dict[str, torch.Tensor | None],
    output_b: dict[str, torch.Tensor | None],
    float_type: torch.dtype,
) -> None:
    """Asserts that two model outputs are equal."""
    a_keys = set(output_a.keys())
    b_keys = set(output_b.keys())
    assert a_keys == b_keys, f"Keys in outputs do not match: {a_keys.symmetric_difference(b_keys)}"

    if float_type == torch.float64:
        rtol = 1e-5
        atol = 1e-8
    else:
        rtol = 1e-3
        atol = 1e-6

    pass_conds = {}
    for key in a_keys:
        v_a, v_b = output_a[key], output_b[key]
        if v_a is None and v_b is None:
            continue
        assert (
            v_a is not None and v_b is not None
        ), f"Key '{key}' is None in one dict but not the other"

        if v_a.shape != v_b.shape:
            LOG.warning(
                f"Key '{key}' has different shapes: {v_a.shape} vs {v_b.shape}. "
                "Attempting to permute dimensions or squeeze if applicable."
            )
            if (
                v_a.ndim == 3
                and v_a.shape[0] == v_b.shape[0]
                and v_a.shape[1] == v_b.shape[2]
                and v_a.shape[2] == v_b.shape[1]
            ):
                v_a = v_a.permute(0, 2, 1).contiguous()
            if v_a.shape[-1] == 1:
                v_a = v_a.squeeze(-1)
            if v_b.shape[-1] == 1:
                v_b = v_b.squeeze(-1)

        assert (
            v_a.shape == v_b.shape
        ), f"Key '{key}' has different shapes: {v_a.shape} vs {v_b.shape}"

        if v_a.dtype == torch.bool:
            v_a = v_a.to(torch.float32)
            v_b = v_b.to(torch.float32)

        pass_conds[key] = check_allclose_detailed(
            v_a.detach().cpu().numpy(), v_b.detach().cpu().numpy(), key, rtol=rtol, atol=atol
        )
    for key, pass_cond in pass_conds.items():
        assert pass_cond, f"{key} comparison failed allclose test"
