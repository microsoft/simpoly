# SPDX-License-Identifier: MIT

from simpoly.poly_arena import experiment


def test_load_data() -> None:
    df = experiment.load_data()
    assert df.shape == (130, 6)


def test_for_nans() -> None:
    df = experiment.load_data()

    # Columns with no NaNs
    columns = ["smiles", "end_group_0", "end_group_1", "density"]

    for col in columns:
        assert not df[col].isna().any()
