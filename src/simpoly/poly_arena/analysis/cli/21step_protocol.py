"""21-step protocol Tg via parametric bootstrap of the self-consistent fit.

For each polymer:
  1. Inverse-variance aggregate stages across seeds.
  2. Draw ρ ~ N(ρ_mean, ρ_std) per stage; refit with self_consistent_fit.
  3. Repeat n_samples times (in a process Pool) → Tg mean / std / Student-t SEM.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from simpoly.poly_arena.analysis.bootstrap import bootstrap_tgs
from simpoly.poly_arena.analysis.experimental_values import EXP_TG

LOG = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stages", type=Path, required=True, help="Aggregated stages CSV (per-seed-per-stage rows)"
    )
    ap.add_argument("--out", type=Path, default=Path("out_21step"))
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--plateau-tol", type=float, default=20.0)
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = pd.read_csv(args.stages)
    LOG.info(
        "[%s] %d rows, %d polymers, %d seeds",
        args.stages.name,
        len(data),
        data["poly_id"].nunique(),
        data["seed"].nunique(),
    )

    df = bootstrap_tgs(
        data,
        n_samples=args.n_samples,
        stride=args.stride,
        plateau_tol=args.plateau_tol,
        n_workers=args.n_workers,
        rng_seed=args.rng_seed,
    )
    df["tg_exp_val"] = df["poly_id"].map(EXP_TG)
    df["tg_delta"] = df["tg_mean_cf"] - df["tg_exp_val"]
    df["tg_ae"] = df["tg_delta"].abs()
    df = df.sort_values("poly_id").reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "tg_results.csv", index=False)
    cols = [
        "model",
        "poly_id",
        "tg_mean_cf",
        "tg_std_cf",
        "tg_sem_t68",
        "tg_exp_val",
        "tg_delta",
        "n_ok",
    ]
    print(df[cols].to_string(index=False, float_format="{:.1f}".format))
    ok = df.dropna(subset=["tg_ae"])
    if len(ok):
        print(
            f"MAE = {ok['tg_ae'].mean():.1f} K | "
            f"median AE = {ok['tg_ae'].median():.1f} K | n = {len(ok)}"
        )


if __name__ == "__main__":
    main()
