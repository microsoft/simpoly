import argparse
import logging
from pathlib import Path

import pandas as pd

from simpoly.poly_arena.analysis.experimental_values import EXP_TG
from simpoly.poly_arena.analysis.pipeline import fit_dataset


def _annotate(agg: pd.DataFrame) -> pd.DataFrame:
    agg["tg_exp_val"] = agg["poly_id"].map(EXP_TG)
    agg["tg_delta"] = agg["tg_mean_cf"] - agg["tg_exp_val"]
    agg["tg_ae"] = agg["tg_delta"].abs()
    return agg


def run(stages_csv: Path, out_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(stages_csv)
    print(
        "[%s] %d rows, %d polymers, %d seeds",
        stages_csv.name,
        len(data),
        data["poly_id"].nunique(),
        data["seed"].nunique(),
    )
    per_seed, agg = fit_dataset(data)
    agg = _annotate(agg).sort_values("poly_id").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "tg_per_seed.csv", index=False)
    agg.to_csv(out_dir / "tg_results.csv", index=False)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out_cooling"))
    ap.add_argument(
        "--inputs",
        nargs="*",
        required=True,
        help="Override input CSVs (default: 100ps + 150ps in revised_tg_analysis/data)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inputs = {Path(p).stem: Path(p) for p in args.inputs}
    for tag, csv in inputs.items():
        agg = run(csv, args.out / tag)
        cols = [
            c
            for c in [
                "model",
                "poly_id",
                "tg_mean_cf",
                "tg_std_cf",
                "tg_exp_val",
                "tg_delta",
                "n_seeds",
            ]
            if c in agg.columns
        ]
        print(f"\n=== {tag} ({csv.name}) ===")
        print(agg[cols].to_string(index=False, float_format="{:.1f}".format))
        ok = agg.dropna(subset=["tg_ae"])
        if len(ok):
            print(
                f"MAE = {ok['tg_ae'].mean():.1f} K | "
                f"median AE = {ok['tg_ae'].median():.1f} K | n = {len(ok)}"
            )


if __name__ == "__main__":
    main()
