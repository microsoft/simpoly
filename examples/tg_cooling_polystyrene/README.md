# MD Example: Polystyrene Tg cooling

## Files

| File | Size | Purpose |
|---|---|---|
| `final_npt.restart` | 865 KB | LAMMPS binary restart — equilibrated start configuration at 533 K / 1 atm |
| `in.tg_cooling`     | 5.5 KB | LAMMPS input A |

## Run with the local Docker image

Build `simpoly:latest` once (see `docker/README.md`) and make
sure `checkpoints/vivace_v0.1.mliap.pt` exists at the repo root (it ships in
this repository). Then, from the repository root:

```bash
docker run --gpus all --rm \
    -v "$PWD/checkpoints:/checkpoints:ro" \
    -v "$PWD/src:/src:ro" \
    -v "$PWD/examples/tg_cooling_polystyrene:/work" \
    -v "$PWD/.torchinductor:/tmp/torchinductor" \
    -e PYTHONPATH=/src \
    -w /work \
    simpoly \
    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.tg_cooling
```

## Notes

- The full protocol is **17 × 300 000 = 5.1 M MD steps** 
  For a fast test, edit
  `in.tg_cooling` and lower the `run` counts (e.g. `run 1000`) or delete
  later cooling stages.
- `pair_coeff * * C C C H` reflects the 4-type element mapping used at
  training time; do not change the order.
- Restart files are LAMMPS-version-sensitive. They were written by
  `patch_22Jul2025` (the version pinned in `docker/Dockerfile`); use the
  same image to read them.
