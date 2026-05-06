"""Per-seed Tg fit + Student-t aggregation across seeds.

One pass over (model, poly_id, seed) groups. Each seed yields a flat row
with the popt unpacked into columns so downstream plotting can read them
straight from CSV, no in-memory dict shuffling between fit and plot.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .fitting import self_consistent_fit

LOG = logging.getLogger(__name__)

_POPT_COLS = ("popt_Tg", "popt_rho0", "popt_a", "popt_b", "popt_c")


def _student_t_sem(values: np.ndarray, conf: float = 0.68) -> float:
    """Student-t half-width on the sample mean. NaN for n<2."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return float("nan")
    s = float(np.std(v, ddof=1))
    if s == 0:
        return 0.0
    t_crit = float(student_t.ppf(0.5 + conf / 2.0, df=n - 1))
    return t_crit * s / np.sqrt(n)


def _empty_per_seed_row(model, poly_id, seed, drop_reason: str) -> dict:
    row = {
        "model": model,
        "poly_id": poly_id,
        "seed": seed,
        "tg": float("nan"),
        "tg0_selected": float("nan"),
        "residual": float("nan"),
        "chosen_basis": f"DROP:{drop_reason}",
        "plateau_lo": float("nan"),
        "plateau_hi": float("nan"),
        "plateau_fit_tg": float("nan"),
        "drop_reason": drop_reason,
    }
    for c in _POPT_COLS:
        row[c] = float("nan")
    return row


def fit_dataset(
    data: pd.DataFrame,
    *,
    stride: int = 10,
    plateau_tol: float = 20.0,
    polymers_to_skip: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "seed" not in data.columns:
        raise ValueError("stages CSV must contain a 'seed' column")
    skip = set(polymers_to_skip or ())

    rows: list[dict] = []
    for (model, poly_id, seed), grp in data.groupby(["model", "poly_id", "seed"]):
        if poly_id in skip:
            continue
        grp = grp.sort_values("temperature_mean")
        T = grp["temperature_mean"].to_numpy(dtype=float)
        rho = grp["density_mean"].to_numpy(dtype=float)

        res = self_consistent_fit(T, rho, stride=stride, plateau_tol=plateau_tol)
        tg_val = res.get("tg")
        if tg_val is None or not np.isfinite(tg_val):
            row = _empty_per_seed_row(model, poly_id, seed, res.get("drop_reason", ""))
        else:
            popt = [res["tg"], res["rho0"], res["a"], res["b"], res["c"]]
            row = {
                "model": model,
                "poly_id": poly_id,
                "seed": seed,
                "tg": float(res["tg"]),
                "tg0_selected": float(res["tg0"]),
                "residual": float("nan"),
                "chosen_basis": res["chosen_basis"],
                "plateau_lo": res["plateau_lo"],
                "plateau_hi": res["plateau_hi"],
                "plateau_fit_tg": res["plateau_fit_tg"],
                "drop_reason": "",
            }
            for c, v in zip(_POPT_COLS, popt):
                row[c] = float(v)
        rows.append(row)

    per_seed_df = pd.DataFrame(rows)
    agg_df = aggregate_seed_tgs(per_seed_df)
    return per_seed_df, agg_df


def aggregate_seed_tgs(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows to per-polymer mean/std/Student-t SEM.

    Picks the median-Tg seed as the representative for diagnostic columns
    (popt for plotting, tg0_selected, plateau_*).
    """
    out = []
    if per_seed_df.empty:
        return pd.DataFrame(out)
    for (model, poly_id), grp in per_seed_df.groupby(["model", "poly_id"]):
        ok = grp.dropna(subset=["tg"])
        n = len(ok)
        n_failed = len(grp) - n
        if n == 0:
            row = {
                "model": model,
                "poly_id": poly_id,
                "tg_mean_cf": float("nan"),
                "tg_std_cf": float("nan"),
                "tg_sem_t68": float("nan"),
                "tg_median_cf": float("nan"),
                "n_seeds": 0,
                "n_seeds_failed": int(n_failed),
                "tg0_selected": float("nan"),
                "chosen_basis": "all_seeds_failed",
            }
            for c in _POPT_COLS:
                row[c] = float("nan")
            out.append(row)
            continue
        tgs = ok["tg"].to_numpy(dtype=float)
        median = float(np.median(tgs))
        rep = ok.iloc[int(np.argmin(np.abs(tgs - median)))]
        row = {
            "model": model,
            "poly_id": poly_id,
            "tg_mean_cf": float(np.mean(tgs)),
            "tg_std_cf": float(np.std(tgs, ddof=1)) if n >= 2 else float("nan"),
            "tg_sem_t68": _student_t_sem(tgs, conf=0.68),
            "tg_median_cf": median,
            "n_seeds": int(n),
            "n_seeds_failed": int(n_failed),
            "chosen_basis": rep.get("chosen_basis", ""),
        }
        for c in _POPT_COLS:
            row[c] = float(rep[c])
        out.append(row)
    return pd.DataFrame(out)


def compute_metrics(df_tgs: pd.DataFrame, exclude: set[str]) -> dict:
    """MAE, median AE, χ²_ν over the fitted Tg table after dropping ``exclude``."""
    sub = df_tgs.dropna(subset=["tg_exp_val", "tg_mean_cf"]).copy()
    sub = sub[~sub["poly_id"].isin(exclude)]
    if sub.empty:
        return {"mae": float("nan"), "median": float("nan"), "chi2": None, "n": 0}
    mae = float(sub["tg_ae"].dropna().mean())
    median = float(sub["tg_ae"].dropna().median())
    sub_z = sub.dropna(subset=["tg_std_cf"])
    sub_z = sub_z[sub_z["tg_std_cf"] > 0]
    chi2 = None
    if not sub_z.empty:
        z = (sub_z["tg_delta"] / sub_z["tg_std_cf"]).values
        chi2 = float(np.mean(z**2))
    return {"mae": mae, "median": median, "chi2": chi2, "n": int(len(sub))}


def popt_from_row(row) -> list[float] | None:
    """Pull popt back out of an agg / per-seed row, or None if missing."""
    try:
        vals = [float(row[c]) for c in _POPT_COLS]
    except (KeyError, TypeError):
        return None
    if any(not np.isfinite(v) for v in vals):
        return None
    return vals
