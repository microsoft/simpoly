ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Single source of truth for the python environment is /environment.yml.
# Use mamba (shipped with miniforge) for fast dependency solving.
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/simpoly/bin:${PATH}"

ARG LAMMPS_VERSION=stable_22Jul2025_update4
ARG KOKKOS_CUDA_ARCHS=AMPERE80
RUN git clone --depth 1 --branch ${LAMMPS_VERSION} \
        https://github.com/lammps/lammps.git /tmp/lammps \
    && mkdir /tmp/lammps/build \
    && cd /tmp/lammps/build \
    && TORCH_CMAKE=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    && TORCH_ABI=$(python -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))') \
    && ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}" \
    && echo "==> torch CXX11 ABI=${TORCH_ABI}; KOKKOS_CUDA_ARCHS=${KOKKOS_CUDA_ARCHS}" \
    && KOKKOS_ARCH_FLAGS="" \
    && for a in $(echo "${KOKKOS_CUDA_ARCHS}" | tr ';,' ' '); do \
            KOKKOS_ARCH_FLAGS="${KOKKOS_ARCH_FLAGS} -DKokkos_ARCH_${a}=ON"; \
       done \
    && export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
    && ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    # mpicxx must wrap nvcc_wrapper so device code is compiled with nvcc while
    # MPI symbols are still resolved. OMPI_CXX is OpenMPI's mechanism for that.
    && export OMPI_CXX=/tmp/lammps/lib/kokkos/bin/nvcc_wrapper \
    && cmake ../cmake \
        -D CMAKE_INSTALL_PREFIX=/usr/local/lammps \
        -D CMAKE_BUILD_TYPE=Release \
        -D BUILD_SHARED_LIBS=yes \
        -D BUILD_MPI=yes \
        -D BUILD_OMP=yes \
        -D PKG_PYTHON=on \
        -D PKG_ML-IAP=on \
        -D PKG_ML-SNAP=on \
        -D MLIAP_ENABLE_PYTHON=on \
        -D PKG_MOLECULE=on \
        -D PKG_EXTRA-MOLECULE=on \
        -D PKG_KSPACE=on \
        -D PKG_CLASS2=on \
        -D PKG_CORESHELL=on \
        -D PKG_DIPOLE=on \
        -D PKG_DRUDE=on \
        -D PKG_MC=on \
        -D PKG_QEQ=on \
        -D PKG_REAXFF=on \
        -D PKG_KOKKOS=on \
        -D Kokkos_ENABLE_CUDA=ON \
        -D Kokkos_ENABLE_OPENMP=ON \
        -D FFT_KOKKOS=KISS \
        ${KOKKOS_ARCH_FLAGS} \
        -D CMAKE_CXX_COMPILER=mpicxx \
        -D CMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
        -D PYTHON_EXECUTABLE=/opt/conda/envs/simpoly/bin/python \
        -D CMAKE_CXX_FLAGS="${ABI_FLAG}" \
        -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -D CMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -D KOKKOSCORE_CXX_FLAGS="${ABI_FLAG}" \
        -D KOKKOS_ALL_COMPILE_OPTIONS="${ABI_FLAG}" \
    && grep -E '^(FFT_KOKKOS|Kokkos_ENABLE_CUDA|Kokkos_ARCH_|BUILD_MPI)' CMakeCache.txt \
    && make -j"$(nproc)" \
    && make install \
    && make install-python \
    && rm -rf /tmp/lammps

# Strip debug symbols from LAMMPS binaries and shared libs.
RUN strip --strip-unneeded /usr/local/lammps/bin/* /usr/local/lammps/lib/*.so* 2>/dev/null || true

# Drop pyc caches and pip metadata that ballooned during the install.
RUN find /opt/conda -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true; \
    find /opt/conda -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true; \
    rm -rf /root/.cache /opt/conda/pkgs

# =====================================================================
# Stage 2 — runtime
# =====================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# build-essential (gcc/g++) is required at runtime so torch.compile's
# Inductor backend can compile generated C++ kernel wrappers. Without it,
# `torch.compile(..., mode="reduce-overhead")` silently falls back to eager.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda/envs/simpoly /opt/conda/envs/simpoly
COPY --from=builder /usr/local/lammps      /usr/local/lammps
# CUDA driver stub — kept in a dedicated dir that is NOT on
# LD_LIBRARY_PATH at runtime so it cannot shadow the real libcuda.so.1
# that nvidia-container-runtime mounts. Only used for the build-time
# `dlopen` sanity check below.
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda_stub/libcuda.so.1

ENV PATH="/usr/local/lammps/bin:/opt/conda/envs/simpoly/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lammps/lib:${LD_LIBRARY_PATH}"

# cuequivariance_ops_torch needs libnvrtc.so.13 from the pip-bundled NVRTC.
ENV LD_LIBRARY_PATH="/opt/conda/envs/simpoly/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}"

# torch>=2.11 eagerly initialises a dynamo disk cache on `import torch` and
# calls getpass.getuser() during it. When the container is run with
# `-u $(id -u)` and that uid has no /etc/passwd entry, getpass raises
# KeyError. getpass.getuser() honours $USER ahead of pwd.getpwuid(), so
# pre-setting USER + a writable cache dir makes the import uid-agnostic
# without any Python-side workaround in vivace.mliap.
ENV USER=simpoly
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# Allow `mpirun` to run as root inside the container (typical for docker run).
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /app/simpoly

# Build-time sanity. No GPU during docker build, so we point the
# loader at the cuda stub for this single command (the stub satisfies
# `dlopen("libcuda.so.1")` but is not on LD_LIBRARY_PATH at runtime).
RUN LD_LIBRARY_PATH="/opt/cuda_stub:${LD_LIBRARY_PATH}" \
    python -c "import torch; from lammps import lammps; \
lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none']); \
lmp.close(); \
print('lammps + torch instantiate OK')"

CMD ["bash"]
