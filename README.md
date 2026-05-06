# SimPoly: Simulation of Polymers with Machine Learning Force Fields Derived from First Principles

[![arXiv](https://img.shields.io/badge/arXiv-2510.13696-b31b1b.svg)](https://arxiv.org/abs/2510.13696)

Companion repository to the paper. It provides the implementation of the **Vivace** machine learning force field, tools for working with the **PolyPack** and **PolyDiss** datasets, and analysis code to reproduce key results.

## Installation

### Recommended: conda/mamba environment

The pinned, tested environment lives in [`environment.yml`](environment.yml) and matches the docker image:

```bash
mamba env create -f environment.yml
mamba activate simpoly
```

`src/` is already on `sys.path` for `pytest` (see `pyproject.toml`), so no editable install is needed to run the tests.

**LAMMPS is not part of the conda environment** and must be built separately against the env's PyTorch/Kokkos toolchain. See [`docker/Dockerfile`](docker/Dockerfile) for the exact CMake invocation (packages, ABI flags, Kokkos arch, MPI wrapper) and [`docker/README.md`](docker/README.md) for build args and notes on the reproducible CUDA 13 image bundling the conda env, LAMMPS, and Kokkos.

### Advanced: pip install

Editable install with optional extras for the Vivace ML stack (`vivace`), CUDA-13 GPU wheels (`cuda13`), and dev tooling (`dev`):

```bash
pip install -e '.[vivace,cuda13,dev]'
```

The conda environment pins matching versions; pip resolution may differ.

## Data & model artifacts

- **Datasets:** [microsoft/simpoly on Hugging Face](https://huggingface.co/microsoft/simpoly)
- **Vivace checkpoints:**
  - [`vivace_v0.1.pt`](/checkpoints/vivace_v0.1.pt), the ASE / Python loader (`vivace.deploy.load_model`, `vivace.calculator.MLFFCalculator`)
  - [`vivace_v0.1.mliap.pt`](/checkpoints/vivace_v0.1.mliap.pt), the LAMMPS loader (`pair_style mliap unified` via `vivace.mliap`)

## PolyData

See the [poly_data notebook](notebooks/poly_data.ipynb) for an introduction to the **PolyPack** and **PolyDiss** datasets.

## PolyArena

### Reference experimental values

See the [poly_arena_experiment notebook](notebooks/poly_arena_experiment.ipynb) for the reference room-temperature densities and glass transition temperatures.

### Generating LAMMPS inputs

Generate the starting configuration and input files with [`run.py`](src/simpoly/poly_arena/simulation/run.py) (use `--help` for all options):

```bash
python src/simpoly/poly_arena/simulation/run.py \
    --directory mysim/ --poly-id PS --temp-k 350 \
    --model-type mlff --mlff-path path/to/vivace_v0.1.mliap.pt
```

### Running the simulations

The generated `21steps.in` is a standard LAMMPS input that loads Vivace via `pair_style mliap unified`. See the [LAMMPS quick start](#lammps-quick-start) below for the input template and the required `lmp` invocation with multi-GPU / Kokkos flags.

To obtain the density-temperature data needed to derive Tg, submit several simulations with different `--temp-k` values.

## Vivace

**Requirements:** NVIDIA GPU with CUDA. cuequivariance kernels are
mandatory; there is no CPU fallback.

### ASE quick start

```python
from ase.io import read
from simpoly.vivace.calculator import MLFFCalculator

# pp_*.lmps atom-type → atomic number (type 1 = C, type 2 = H).
Z_OF_TYPE = {1: 6, 2: 1}

atoms = read("tests/vivace/data/pp_274.lmps", format="lammps-data", Z_of_type=Z_OF_TYPE)
atoms.calc = MLFFCalculator(model_path="checkpoints/vivace_v0.1.pt")

energy = float(atoms.get_potential_energy())          # eV
forces = atoms.get_forces()                            # eV / Å, (N, 3)
stress_voigt = atoms.get_stress(voigt=True)            # eV / Å^3, length-6 Voigt
```

### LAMMPS quick start

Adapted from [`tests/vivace/test_lammps_mliap.py`](tests/vivace/test_lammps_mliap.py). Vivace loads as a LAMMPS `mliap unified` pair style; the docker image provides the `lmp` binary.

Minimal `in.lmp`:

```text
units           metal
atom_style      atomic
boundary        p p p
newton          on

read_data       tests/vivace/data/pp_274.lmps

mass 1 12.011  # C
mass 2  1.008  # H

pair_style      mliap unified checkpoints/vivace_v0.1.mliap.pt 0
pair_coeff      * * C H

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz
run             0
```

Run it on a single GPU via Kokkos:

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp -log log.lammps
```

For a multi-GPU production run (e.g. 4 GPUs over MPI):

```bash
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos neigh half -in 21steps.in
```

## Running the tests

```bash
pytest tests/                  # full suite
pytest tests/ -m "not gpu"     # skip GPU-only tests
pytest tests/vivace/ -m gpu    # GPU-only end-to-end checks
```
