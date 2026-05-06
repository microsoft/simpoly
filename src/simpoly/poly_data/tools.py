# SPDX-License-Identifier: MIT

import pickle
import typing as ty

import ase
import ase.calculators.singlepoint as ase_sp_calculators
import ase.stress
import lmdb
import numpy as np
import numpy.typing as npt

T = ty.TypeVar("T", covariant=False)


class LmdbReader(ty.Generic[T]):
    def __init__(self, db_path: str) -> None:
        super().__init__()
        self.db_path = db_path

        self.data: lmdb.Environment = lmdb.open(
            path=self.db_path,
            subdir=False,
            sync=False,
            writemap=False,
            meminit=False,
            map_async=False,
            create=False,
            readonly=True,
            lock=False,
        )
        self.is_open = True

    @staticmethod
    def _generate_key(key: int) -> bytes:
        return key.to_bytes(length=8, byteorder="little")

    def __getitem__(self, index: int) -> T:
        assert self.is_open
        if not isinstance(index, int):
            raise IndexError(f"{self.__class__.__name__} expects int index")

        with self.data.begin(write=False) as tx:
            key = self._generate_key(index)
            buf = tx.get(key)
            if not buf:
                raise IndexError()
            val: T = pickle.loads(buf)
            return val

    def __len__(self) -> int:
        return self.data.stat()["entries"]  # type: ignore

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        # If the lmdb file cannot be opened in the initializer, the attribute `is_open` will not be set.
        # For this reason, we check if `is_open` is an attribute of the object before checking its value.
        if not hasattr(self, "is_open") or not self.is_open:
            return

        self.data.close()
        self.is_open = False


DataArray: ty.TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.int32] | npt.NDArray[np.bool_]
ArrayDictReader: ty.TypeAlias = LmdbReader[dict[str, DataArray]]


def array_dict_to_atoms(d: dict[str, DataArray]) -> ase.Atoms:
    atoms = ase.Atoms(  # type: ignore
        numbers=d["atomic_numbers"],
        positions=d["pos"],
        pbc=d["pbc"] if "pbc" in d else None,
        cell=d["cell"] if "cell" in d else None,
    )

    # Store energy, forces, and stress
    results = {}
    if "energy" in d:
        results["energy"] = d["energy"].squeeze(0)

    if "forces" in d:
        results["forces"] = d["forces"]

    # ASE uses stress, not virial
    if "virial" in d:
        results["stress"] = ase.stress.full_3x3_to_voigt_6_stress(  # type: ignore
            -1 * d["virial"] / atoms.get_volume()  # type: ignore
        )

    atoms.calc = ase_sp_calculators.SinglePointCalculator(atoms, **results)  # type: ignore

    return atoms
