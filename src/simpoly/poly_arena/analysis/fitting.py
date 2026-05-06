"""Patrone curve_fit wrappers: single-shot fit, plateau picker, bootstrap worker."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

_FIT_BOUNDS = (-np.inf, [np.inf, np.inf, np.inf, np.inf, 3.0])
_FITTING_KW = dict(method="trf", maxfev=1003600, bounds=_FIT_BOUNDS)


def patrone_equation(
    T: npt.NDArray[np.float64],
    Tg: float,
    rho0: float,
    a: float,
    b: float,
    c: float,
) -> npt.NDArray[np.float64]:
    """Patrone-form rho(T) (Patrone et al. 2017, eq. 5)."""
    d = T - Tg
    return rho0 - a * d - b * (0.5 * d + np.sqrt(0.25 * d**2 + np.exp(c)))


def wrapped_patrone_fit(
    T: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    tg0: float,
) -> tuple[float, float, float, float, float, float, str]:
    """Single deterministic curve_fit at a given tg0 initial guess.

    Returns ``(popt, mse, err)``. On failure ``popt`` is ``None``,
    ``mse`` is ``inf`` and ``err`` is the exception string.
    """
    p0 = [tg0, float(np.mean(rho)), 1e-3, 3e-3, 2.0]
    try:
        popt, _ = curve_fit(patrone_equation, T, rho, p0=p0, **_FITTING_KW)
        tg, rho0, a, b, c = popt
        mse = float(np.mean((rho - patrone_equation(T, *popt)) ** 2))
        return tg, rho0, a, b, c, mse, ""
    except Exception as e:  # noqa: BLE001  curve_fit raises many shapes
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("inf"),
            str(e),
        )


def _bootstrap_worker(task) -> tuple[Any, list[float]]:
    """Parametric bootstrap for one polymer (pickled → Pool workers).

    task = (key, T, rho, rho_std, p0, n_samples, seed); returns (key, [tgs]).
    """
    key, T, rho, rho_std, p0, n_samples, seed = task
    rng = np.random.default_rng(seed)
    tgs: list[float] = []
    for _ in range(n_samples):
        rho_b = rng.normal(rho, rho_std)
        try:
            pb, _ = curve_fit(patrone_equation, T, rho_b, p0=p0, **_FITTING_KW)
            tgs.append(float(pb[0]))
        except Exception:
            pass
    return key, tgs


def _find_plateau(
    fts: np.ndarray,
    tg0s: np.ndarray,
    *,
    plateau_tol: float,
    sc_thresh: float,
    T_lo: float,
    T_hi: float,
    require_sc: bool,
    longest: bool,
) -> tuple[int, int]:
    """Walk tg0 low→high; return (lo, hi) idx of first / longest plateau, or (-1, -1)."""
    n_v = len(fts)
    best_lo_, best_hi_, best_count_ = -1, -1, -1
    for i in range(n_v):
        last_j = i
        j = i + 1
        while j <= n_v:
            w = slice(i, j)
            ok = (fts[w].max() - fts[w].min()) <= plateau_tol
            if ok and require_sc:
                ok = bool(np.all(np.abs(fts[w] - tg0s[w]) <= sc_thresh))
            if not ok:
                break
            last_j = j
            j += 1
        count = last_j - i
        if count < 2:
            continue
        med = float(np.median(fts[i:last_j]))
        if not (T_lo <= med <= T_hi):
            continue
        if longest:
            if count > best_count_:
                best_lo_, best_hi_, best_count_ = i, last_j - 1, count
        else:
            return i, last_j - 1
    return best_lo_, best_hi_


def self_consistent_fit(
    T: np.ndarray,
    rho: np.ndarray,
    *,
    T_min: int = 0,
    T_max: int = 1000,
    stride: int = 5,
    plateau_tol: float = 10.0,
) -> dict[str, Any]:
    """Sweep tg0 ∈ [T_min, T_max] step ``stride``; return the plateau-picked best fit.

    Returns dict with keys::

        tg              -- candidate dict (tg0, tg, residual, popt, self_consistent)
                           or None if no fit converged
        chosen_basis    -- string tag describing how the pick was made
        plateau_lo/hi   -- tg0 bounds of the chosen plateau (NaN on fallback)
        plateau_fit_tg  -- median fit_tg inside the plateau (NaN on fallback)
        drop_reason     -- non-empty when ``tg`` is None
    """
    if len(T) < 5:
        reason = f"too_few_stages_n={len(T)}"
        return {
            "tg": None,
            "chosen_basis": "DROP:" + reason,
            "plateau_lo": float("nan"),
            "plateau_hi": float("nan"),
            "plateau_fit_tg": float("nan"),
            "drop_reason": reason,
        }

    sc_thresh = 2.0 * stride
    valid_fits: list[dict[str, Any]] = []
    for tg0 in range(T_min, T_max + stride, stride):
        tg, rho0, a, b, c, mse, err = wrapped_patrone_fit(T, rho, float(tg0))
        if err:
            continue
        if not np.isfinite(tg):
            continue
        valid_fits.append(
            {
                "tg0": tg0,
                "tg": tg,
                "residual": mse,
                "rho0": rho0,
                "a": a,
                "b": b,
                "c": c,
                "self_consistent": abs(tg - tg0) <= stride,
            }
        )

    if not valid_fits:
        return {
            "chosen_basis": "no_valid_fits",
            "tg": float("nan"),
            "tg0": float("nan"),
            "rho0": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "drop_reason": "no_valid_fits",
        }

    valid_sorted = sorted(valid_fits, key=lambda c: c["tg0"])
    fts = np.array([c["tg"] for c in valid_sorted])
    tg0s = np.array([c["tg0"] for c in valid_sorted])
    T_lo, T_hi = float(T.min()), float(T.max())

    best_lo, best_hi = _find_plateau(
        fts,
        tg0s,
        plateau_tol=plateau_tol,
        sc_thresh=sc_thresh,
        T_lo=T_lo,
        T_hi=T_hi,
        require_sc=True,
        longest=False,
    )
    sel_tag = "hybrid_sc"
    if best_lo < 0:
        best_lo, best_hi = _find_plateau(
            fts,
            tg0s,
            plateau_tol=plateau_tol,
            sc_thresh=sc_thresh,
            T_lo=T_lo,
            T_hi=T_hi,
            require_sc=False,
            longest=True,
        )
        sel_tag = "hybrid_longest"

    if best_lo >= 0:
        plateau_lo = float(valid_sorted[best_lo]["tg0"])
        plateau_hi = float(valid_sorted[best_hi]["tg0"])
        plateau_fit_tg = float(np.median(fts[best_lo : best_hi + 1]))
        chosen_basis = (
            f"{sel_tag}_upper_edge[{plateau_lo:.0f},"
            f"{plateau_hi:.0f}]_fit_tg≈{plateau_fit_tg:.0f}"
        )
        tail = valid_sorted[max(best_lo, best_hi - 2) : best_hi + 1]
        best = min(tail, key=lambda c: c["residual"])
    else:
        best = min(valid_fits, key=lambda c: c["residual"])
        chosen_basis = "fallback_min_residual_no_plateau"
        plateau_lo = plateau_hi = plateau_fit_tg = float("nan")

    return {
        "tg": best["tg"],
        "tg0": best["tg0"],
        "rho0": best["rho0"],
        "a": best["a"],
        "b": best["b"],
        "c": best["c"],
        "chosen_basis": chosen_basis,
        "plateau_lo": plateau_lo,
        "plateau_hi": plateau_hi,
        "plateau_fit_tg": plateau_fit_tg,
        "drop_reason": "",
    }
