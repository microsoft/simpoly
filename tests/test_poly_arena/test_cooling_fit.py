"""Smoke test for the cooling per-seed Tg pipeline on synthetic data.

Generates a Patrone-form rho(T) curve with known Tg, perturbs each seed with
small Gaussian noise, runs `fit_dataset`, and asserts the recovered Tg sits
within a few K of ground truth.
"""

import numpy as np
import pandas as pd
import pytest

from simpoly.poly_arena.analysis.fitting import patrone_equation
from simpoly.poly_arena.analysis.pipeline import _student_t_sem, fit_dataset


def _synthetic_stages(true_tg=400.0, n_stages=15, n_seeds=3, noise=0.005, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    Ts = np.linspace(200, 600, n_stages)
    rows = []
    for seed in range(n_seeds):
        rho = patrone_equation(Ts, true_tg, rho0=1.0, a=5e-4, b=5e-4, c=2.0)
        rho_noisy = rho + rng.normal(0, noise, size=Ts.shape)
        for i, (T, r) in enumerate(zip(Ts, rho_noisy)):
            rows.append(
                dict(
                    model="TEST",
                    poly_id="DUMMY",
                    seed=seed,
                    delta_T=20.0,
                    temperature_mean=T,
                    temperature_std=2.0,
                    density_mean=r,
                    density_std=0.001,
                )
            )
    return pd.DataFrame(rows)


def test_fit_dataset_recovers_known_tg():
    df = _synthetic_stages(true_tg=400.0)
    per_seed, agg = fit_dataset(df, stride=10, plateau_tol=20.0)
    assert len(agg) == 1, "should produce one row per (model, poly_id)"
    row = agg.iloc[0]
    assert row["n_seeds"] == 3
    assert abs(row["tg_mean_cf"] - 400.0) < 15, f"Tg recovered at {row['tg_mean_cf']}"
    seed_tgs = per_seed["tg"].dropna().values
    assert len(seed_tgs) == 3
    assert seed_tgs.std(ddof=1) < 10, f"seed scatter {seed_tgs.std(ddof=1)} too large"


def test_fit_dataset_skips_polymers_in_skip_list():
    df1 = _synthetic_stages(true_tg=400.0)
    df2 = _synthetic_stages(true_tg=420.0)
    df2["poly_id"] = "OTHER"
    df = pd.concat([df1, df2], ignore_index=True)
    per_seed, agg = fit_dataset(df, stride=10, plateau_tol=20.0, polymers_to_skip={"DUMMY"})
    assert set(agg["poly_id"]) == {"OTHER"}, "DUMMY should be skipped"
    assert "DUMMY" not in set(per_seed["poly_id"])


def test_student_t_sem_for_n3():
    val = _student_t_sem(np.array([390.0, 400.0, 410.0]), conf=0.68)
    assert abs(val - 7.57) < 0.5, f"got {val}"


def test_student_t_sem_n_lt_2_returns_nan():
    assert np.isnan(_student_t_sem(np.array([400.0]), conf=0.68))
    assert np.isnan(_student_t_sem(np.array([]), conf=0.68))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
