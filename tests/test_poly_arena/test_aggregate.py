# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from simpoly.poly_arena.analysis.aggregate import (
    aggregate_measurements,
    aggregate_simulations,
)


def test_aggregate_measurements_basic():
    values = [1.0, 2.0, 3.0]
    stds = [1.0, 2.0, 1.0]
    combined_val, combined_std = aggregate_measurements(values, stds)
    weights = 1 / np.square(stds)
    expected_val = np.sum(weights * values) / np.sum(weights)
    expected_std = 1 / np.sqrt(np.sum(weights))
    assert np.isclose(combined_val, expected_val)
    assert np.isclose(combined_std, expected_std)


def test_aggregate_measurements_single():
    values = [5.0]
    stds = [0.5]
    combined_val, combined_std = aggregate_measurements(values, stds)
    assert combined_val == 5.0
    assert np.isclose(combined_std, 0.5)


def test_aggregate_simulations_mean_and_std():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    cols = [("val", None)]
    result = aggregate_simulations(df, cols)
    assert np.isclose(result["val"], 2.0)
    assert np.isclose(result["val_std"], np.std([1.0, 2.0, 3.0], ddof=1))
    assert result["n"] == 3


def test_aggregate_simulations_weighted():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "std": [0.1, 0.2, 0.1]})
    cols = [("val", "std")]
    result = aggregate_simulations(df, cols)
    weights = 1 / np.square(df["std"])
    expected_val = np.sum(weights * df["val"]) / np.sum(weights)
    expected_std = 1 / np.sqrt(np.sum(weights))
    assert np.isclose(result["val"], expected_val)
    assert np.isclose(result["std"], expected_std)
    assert "val_std" not in result.index
    assert result["n"] == 3


def test_aggregate_simulations_nan_handling():
    df = pd.DataFrame({"val": [np.nan, 2.0], "std": [np.nan, 0.2]})
    cols = [("val", "std")]
    result = aggregate_simulations(df, cols)
    assert result["val"] == 2.0
    assert result["std"] == 0.2
    assert "val_std" not in result.index
    assert result["n"] == 2


def test_aggregate_simulations_by():
    df = pd.DataFrame(
        {
            "poly_id": [1, 1, 2, 2, 2],
            "model": ["A", "A", "B", "B", "B"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
            "std": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    cols = [("val", "std")]
    result = aggregate_simulations(df, cols, by=["poly_id", "model"])
    assert set(result["n"]) == {2, 3}
    assert "val" in result.columns
    assert "std" in result.columns
    assert "val_std" not in result.columns
    group = df[(df["poly_id"] == 1) & (df["model"] == "A")]
    weights = 1 / np.square(group["std"])
    expected_val = np.sum(weights * group["val"]) / np.sum(weights)
    expected_std = 1 / np.sqrt(np.sum(weights))
    row = result[(result["poly_id"] == 1) & (result["model"] == "A")].iloc[0]
    assert np.isclose(row["val"], expected_val)
    assert np.isclose(row["std"], expected_std)
