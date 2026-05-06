import pathlib
import shutil
import subprocess

import numpy as np
import pytest
from ase.data import atomic_masses, atomic_numbers


@pytest.mark.parametrize("n_atoms", [274, 7398])
def test_lammps_mliap(
    test_data_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    test_checkpoint_pt: pathlib.Path,
    n_atoms: int,
) -> None:

    data = test_data_dir / f"pp_{n_atoms}.lmps"
    ref = test_data_dir / f"pp_{n_atoms}_lammps_reference.npz"

    # find the lammps binary in the environment (public docker ships ``lmp``;
    # the legacy reference-generation docker shipped ``lmp_cuda``).
    lmp = shutil.which("lmp") or shutil.which("lmp_cuda")
    if lmp is None:
        pytest.skip(
            "lmp/lmp_cuda not found in PATH; ensure you are running inside the correct docker container"
        )

    _n_atoms, energy, force, pressure = run(
        data=data,
        mliap=test_checkpoint_pt,
        lmp=lmp,
        lmp_args=[
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
        ],
        atom_types=["C", "H"],
        work=tmp_path,
    )

    # The reference NPZ key is named ``stress_bar`` for historical reasons but
    # actually stores LAMMPS pressure (px*) at thermo precision. Compare
    # ``pressure`` directly.
    reference = np.load(ref)

    assert _n_atoms == n_atoms
    assert n_atoms == reference["n_atoms"]

    # Energy comes from thermo at full double precision -> tight rtol.
    # Forces / pressure are limited by the LAMMPS dump/thermo print precision
    # (~6 significant figures), so use an absolute tolerance commensurate with
    # the values' magnitudes rather than a fractional one.
    assert np.isclose(
        energy, reference["energy_eV"], rtol=1e-6
    ), f"energy mismatch: {energy} vs {reference['energy_eV']}"
    f_ref = reference["forces_eV_per_A"]
    assert np.allclose(
        force, f_ref, atol=2e-5
    ), f"force mismatch: max|ΔF|={np.abs(force - f_ref).max():.4e} eV/A"
    p_ref = reference["pressure_bar"]
    assert np.allclose(
        pressure, p_ref, atol=1.0, rtol=1e-4
    ), f"pressure mismatch: max|ΔP|={np.abs(pressure - p_ref).max():.4e} bar"


def _masses(atom_types: list[str]) -> str:
    return "\n".join(
        f"mass {i} {atomic_masses[atomic_numbers[s]]:.6f}  # {s}"
        for i, s in enumerate(atom_types, 1)
    )


def _parse_thermo(log_text: str) -> tuple[float, np.ndarray]:
    """First post-header data row: step 0, initial positions."""
    lines = log_text.splitlines()
    hdr_idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Step"))
    data = lines[hdr_idx + 1].split()
    pe = float(data[1])
    pxx, pyy, pzz, pxy, pxz, pyz = (float(x) for x in data[2:8])
    return pe, np.array([pxx, pyy, pzz, pxy, pxz, pyz])


def _parse_dump(dump_text: str, n_atoms: int) -> np.ndarray:
    lines = dump_text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("ITEM: ATOMS"))
    rows = lines[idx + 1 : idx + 1 + n_atoms]
    arr = np.array([[float(x) for x in r.split()] for r in rows])
    arr = arr[arr[:, 0].argsort()]  # sort by id; KOKKOS dumps unsorted
    return arr[:, 1:4]


def run(
    data: pathlib.Path,
    mliap: pathlib.Path,
    lmp: str,
    lmp_args: list[str],
    atom_types: list[str],
    work: pathlib.Path,
) -> tuple[int, float, np.ndarray, np.ndarray]:

    # ".lmps" line 3 is "<n> atoms" per LAMMPS data-file spec
    n_atoms = int(open(data).readlines()[2].split()[0])

    rdir = work / "run"
    rdir.mkdir()
    dump_file = rdir / "forces.dump"
    template = """\
units           metal
atom_style      atomic
boundary        p p p
newton          on

read_data       {data_file}

{masses}

pair_style      mliap unified {mliap_path} 0
pair_coeff      * * {atom_types}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz
dump            f all custom 1 {dump_file} id fx fy fz
run             0
"""
    inp = template.format(
        data_file=str(data),
        masses=_masses(atom_types),
        mliap_path=str(mliap),
        atom_types=" ".join(atom_types),
        dump_file=str(dump_file),
    )
    (rdir / "in.lmp").write_text(inp)
    proc = subprocess.run(
        [lmp, *lmp_args, "-in", "in.lmp", "-log", "log.lammps"],
        cwd=rdir,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = rdir / "log.lammps"
    if not log_path.exists():
        print("STDOUT:", proc.stdout[-2000:])
        print("STDERR:", proc.stderr[-2000:])
        raise SystemExit(f"lmp produced no log (rc={proc.returncode})")
    log = log_path.read_text()
    if proc.returncode != 0:
        print(log[-2000:])
        print("STDERR:", proc.stderr[-500:])
        raise SystemExit("lmp failed run")
    energy, pressure = _parse_thermo(log)
    force = _parse_dump(dump_file.read_text(), n_atoms)
    print(
        f"  run : pe={energy:.10e}  |F|max={np.abs(force).max():.4f} eV/A  P[bar]={pressure[:3].mean():.2f}±{pressure[:3].std():.2f}"
    )

    return n_atoms, energy, force, pressure
