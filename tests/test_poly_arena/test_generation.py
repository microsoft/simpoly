# SPDX-License-Identifier: MIT

import pathlib

import pytest

from simpoly.poly_arena import experiment, generation


@pytest.mark.parametrize(
    "poly_id",
    ["PP", "PAN", "PCTFE"],
)
def test_simple(poly_id: str, tmp_path: pathlib.Path) -> None:
    data = experiment.load_data()

    smiles = data.loc[poly_id, "smiles"]
    end_group_0 = data.loc[poly_id, "end_group_0"]
    end_group_1 = data.loc[poly_id, "end_group_1"]

    assert isinstance(smiles, str)
    assert isinstance(end_group_0, str)
    assert isinstance(end_group_1, str)

    config = generation.Config(
        ru_smiles=smiles,
        first_cap=end_group_0,
        second_cap=end_group_1,
        n_ru_per_chain=2,
        density=0.1,
        temperature=298,
        n_total=100,
        seed=42,
    )

    generation.prepare(config, directory=str(tmp_path))


def test_long_polymer(tmp_path: pathlib.Path) -> None:
    smiles = "*CCC*"  # 9 atoms
    end_groups = ("H*", "H*")  # 1 atom each

    config = generation.Config(
        ru_smiles=smiles,
        first_cap=end_groups[0],
        second_cap=end_groups[1],
        n_ru_per_chain=10,
        density=0.1,
        temperature=298,
        n_total=10,  # too small for 10 chains!
        seed=42,
    )

    with pytest.raises(generation.SegfaultError):
        generation.prepare(config, directory=str(tmp_path))
