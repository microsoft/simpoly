# Building the docker image

The image bundles the `simpoly` conda env (see [`environment.yml`](../environment.yml))
plus a LAMMPS build (Kokkos + ML-IAP + Python coupling) compiled against the
env's PyTorch. See the top-level [README](../README.md) for usage.

```bash
docker build -t simpoly -f docker/Dockerfile .
```

The build context **must** be the repo root.

## Build args

| Arg | Default | Notes |
|---|---|---|
| `KOKKOS_CUDA_ARCHS` | `AMPERE80` | Set to your GPU arch (e.g. `HOPPER90`). |
| `LAMMPS_VERSION` | `stable_22Jul2025_update4` | Any LAMMPS git tag. |
| `CUDA_VERSION` / `UBUNTU_VERSION` | `13.0.3` / `22.04` | Must match a published `nvidia/cuda` tag. |

## Notes

- LAMMPS is built with `-D_GLIBCXX_USE_CXX11_ABI` matching the PyTorch wheel.
  Do not override.
- A build-time sanity check imports `torch` and instantiates `lammps()` against
  the CUDA stub; the build fails fast if the ABI/linkage is wrong.
