# SPDX-License-Identifier: MIT

import pathlib

import pandas as pd


def load_data() -> pd.DataFrame:
    path = pathlib.Path(__file__).parent / "resources" / "data.csv"

    df = pd.read_csv(
        path,
        na_values=["NaN"],
        encoding="utf-8",
        comment="%",  # to avoid conflict with SMILES containing "#"
    )

    df = df.set_index("id").sort_index()

    return df
