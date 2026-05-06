"""Jean-style parametric bootstrap on per-stage aggregated data.

1. Inverse-variance aggregate seeds → one (T, ρ, ρ_std) per stage.
2. For each polymer: draw rho_b ~ N(rho, rho_std), refit Tg via
   :func:`simpoly.poly_arena.analysis.fitting.self_consistent_fit` (full SC sweep per draw).
3. ``mp.Pool`` over polymers; aggregate Tg distribution → mean/std/SEM.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .fitting import patrone_equation, self_consistent_fit

LOG = logging.getLogger(__name__)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse-variance weighted aggregation across seeds per stage."""

    def _one(sub: pd.DataFrame) -> pd.Series:
        out: dict = {"n_seeds": int(len(sub))}
        for v, s in (("temperature_mean", "temperature_std"), ("density_mean", "density_std")):
            vals = sub[v].to_numpy(float)
            stds = sub[s].to_numpy(float)
            m = np.isfinite(vals) & np.isfinite(stds)
            vals, stds = vals[m], stds[m]
            if vals.size == 0:
                out[v] = out[s] = float("nan")
            elif vals.size == 1:
                out[v], out[s] = float(vals[0]), float(stds[0])
            else:
                w = 1.0 / np.maximum(stds, 1e-12) ** 2
                out[v] = float(np.average(vals, weights=w))
                out[s] = float(1.0 / np.sqrt(w.sum()))
        return pd.Series(out)

    return (
        df.groupby(["model", "poly_id", "delta_T"], sort=True)
        .apply(_one, include_groups=False)
        .reset_index()
    )


def _sc_worker(task):
    """One *chunk* of bootstrap samples for one polymer.

    task = (key, T, rho, rho_std, n_samples, stride, plateau_tol, seed)
    Returns (key, [tgs]) for this chunk.
    """
    key, T, rho, rho_std, n_samples, stride, plateau_tol, seed = task
    rng = np.random.default_rng(seed)
    tgs: list[float] = []
    for _ in range(n_samples):
        res = self_consistent_fit(
            T, rng.normal(rho, rho_std), stride=stride, plateau_tol=plateau_tol
        )
        tg = res.get("tg")
        if tg is not None and np.isfinite(tg):
            tgs.append(float(tg))
    return key, tgs


def _student_t_sem(values: np.ndarray, conf: float = 0.68) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    s = float(np.std(v, ddof=1))
    if s == 0:
        return 0.0
    t_crit = float(student_t.ppf(0.5 + conf / 2.0, df=len(v) - 1))
    return t_crit * s / np.sqrt(len(v))


def bootstrap_tgs(
    df: pd.DataFrame,
    *,
    n_samples: int = 500,
    stride: int = 10,
    plateau_tol: float = 20.0,
    n_workers: int | None = None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per-polymer parametric SC bootstrap → mean/std/Student-t SEM Tg.

    Parallelism unit is a *chunk* of bootstrap samples for a single polymer
    (not a whole polymer), so all ``n_workers`` cores stay busy even when
    ``n_polymers < n_workers``.
    """
    agg = aggregate_seeds(df)

    # First collect one (key, T, rho, rho_std) per eligible polymer.
    polymers: list[tuple] = []
    ss = np.random.SeedSequence(rng_seed)
    for ((model, poly_id), grp), child in zip(
        agg.groupby(["model", "poly_id"], sort=True),
        ss.spawn(agg.groupby(["model", "poly_id"]).ngroups),
    ):
        grp = grp.sort_values("temperature_mean")
        T = grp["temperature_mean"].to_numpy(float)
        rho = grp["density_mean"].to_numpy(float)
        rho_std = grp["density_std"].to_numpy(float)
        if len(T) < 5 or not np.all(np.isfinite(rho_std) & (rho_std > 0)):
            LOG.warning(
                "Skipping %s/%s (n=%d, std ok=%s)",
                model,
                poly_id,
                len(T),
                bool(np.all(np.isfinite(rho_std) & (rho_std > 0))),
            )
            continue
        polymers.append(((model, poly_id), T, rho, rho_std, child))

    if not polymers:
        return pd.DataFrame()

    n_workers = max(1, min(n_workers or (os.cpu_count() or 1) - 1, len(polymers) * n_samples))

    # Split each polymer's n_samples into chunks so total task count
    # ~= 4 * n_workers, keeps all cores busy while amortizing pool overhead.
    target_total_tasks = max(n_workers * 4, len(polymers))
    chunks_per_polymer = max(1, target_total_tasks // len(polymers))
    chunks_per_polymer = min(chunks_per_polymer, n_samples)

    tasks = []
    for key, T, rho, rho_std, child in polymers:
        # Distribute n_samples across chunks_per_polymer chunks (last chunk may be larger).
        base, extra = divmod(n_samples, chunks_per_polymer)
        sizes = [base + (1 if i < extra else 0) for i in range(chunks_per_polymer)]
        chunk_seeds = child.spawn(chunks_per_polymer)
        for sz, cseed in zip(sizes, chunk_seeds):
            if sz == 0:
                continue
            tasks.append(
                (key, T, rho, rho_std, sz, stride, plateau_tol, int(cseed.generate_state(1)[0]))
            )

    LOG.info(
        "Bootstrapping %d polymers × %d samples on %d workers (%d tasks, ~%d samples/task)",
        len(polymers),
        n_samples,
        n_workers,
        len(tasks),
        n_samples // chunks_per_polymer,
    )

    boot: dict = {key: [] for key, *_ in polymers}

    if n_workers > 1:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for key, tgs in pool.imap_unordered(_sc_worker, tasks):
                boot[key].extend(tgs)
    else:
        for task in tasks:
            key, tgs = _sc_worker(task)
            boot[key].extend(tgs)

    rows = []
    for key, *_ in polymers:
        model, poly_id = key
        tgs = np.asarray(boot.get(key, []), float)
        n_ok = int(len(tgs))
        rows.append(
            {
                "model": model,
                "poly_id": poly_id,
                "tg_mean_cf": float(np.mean(tgs)) if n_ok else float("nan"),
                "tg_median_cf": float(np.median(tgs)) if n_ok else float("nan"),
                "tg_std_cf": float(np.std(tgs, ddof=1)) if n_ok >= 2 else float("nan"),
                "tg_sem_t68": _student_t_sem(tgs),
                "n_ok": n_ok,
                "n_failed": int(n_samples - n_ok),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["aggregate_seeds", "bootstrap_tgs", "patrone_equation"]
