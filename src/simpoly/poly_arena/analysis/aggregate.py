# SPDX-License-Identifier: MIT

import typing as ty

import numpy as np
import pandas as pd


def aggregate_measurements(values, stds):
    """
    Aggregate measurements using inverse variance weighting: more precise measurements (smaller std) get higher weight.
    """
    values = np.array(values)
    stds = np.array(stds)

    # weights are inverse of variance (1/σ²)
    weights = 1 / (stds**2)

    # weighted average
    combined_value = np.sum(weights * values) / np.sum(weights)

    # combined standard deviation
    combined_std = 1 / np.sqrt(np.sum(weights))

    return combined_value, combined_std


def aggregate_simulations(
    df: pd.DataFrame,
    cols: ty.Sequence[tuple[str, str | None]],
    by: ty.Sequence[str] | None = None,
):
    """
    Aggregate measurements in a DataFrame, optionally grouped by specified columns.
    """

    def _aggregate(df: pd.DataFrame, cols: ty.Sequence[tuple[str, str | None]]) -> pd.Series:
        result: dict[str, int | float] = {"n": len(df)}  # count of elements aggregated

        for val_col, std_col in cols:
            # case: no weights provided, just compute mean and std
            if std_col is None:
                result[val_col] = df[val_col].mean()
                result[f"{val_col}_std"] = df[val_col].std()

            # case: weights provided, use inverse variance weighting
            else:
                # measurement values and associated noise
                values = df[val_col].to_numpy()
                stds = df[std_col].to_numpy()

                # remove NaNs
                mask = ~(np.isnan(values) | np.isnan(stds))
                values = values[mask]
                stds = stds[mask]

                # combine measurements
                if len(values) == 0:
                    result.update({val_col: np.nan, std_col: np.nan})
                elif len(values) == 1:
                    result.update({val_col: values[0], std_col: stds[0]})
                else:
                    combined_val, combined_std = aggregate_measurements(values, stds)
                    result.update(
                        {
                            val_col: combined_val,
                            std_col: combined_std,
                        }
                    )

        return pd.Series(result)

    if by is None:
        return _aggregate(df, cols=cols)

    return df.groupby(by=by).apply(_aggregate, cols=cols, include_groups=False).reset_index()  # type: ignore
