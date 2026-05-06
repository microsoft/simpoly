"""
copy from Schnetpack with small modification of syntax
"""

import logging
import math
from functools import lru_cache
from typing import Tuple

import torch
from sympy.physics.wigner import clebsch_gordan


@lru_cache(maxsize=10)
def sh_indices(
    lmax: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], int]]:
    """Build index arrays for spherical harmonics"""
    ls = torch.arange(0, lmax + 1)
    nls = 2 * ls + 1
    lidx = torch.repeat_interleave(ls, nls)
    midx = torch.cat([torch.arange(-l, l + 1) for l in ls])  # pyright: ignore
    mapping = {(l, m): i for i, (l, m) in enumerate(zip(lidx.numpy(), midx.numpy()))}
    return lidx, midx, mapping


def clebsch_gordan_from_sympy(
    dtype: torch.dtype,
    instructions: list[tuple[int, int, int, int, int, int]],
) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int, int, int, int, int]]]:
    """Generate Clebsch-Gordan coefficients for real spherical harmonics in sparse format

    Returns:
        cg_sparse: vector of non-zeros CG coefficients
        idx_in_1: indices for first set of irreps
        idx_in_2: indices for second set of irreps
        idx_out: indices for output set of irreps
    """
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    lmax1 = max([l1 for l1, _, _, _, _, _ in instructions])
    lmax2 = max([l2 for _, _, l2, _, _, _ in instructions])
    lmax3 = max([l3 for _, _, _, _, l3, _ in instructions])

    # sanity check
    for l1, p1, l2, p2, l3, p3 in instructions:
        assert abs(l1 - l2) <= l3 <= min(l1 + l2, l3)
        assert p1 * p2 == p3  # parity: even -> 1, odd -> -1

    lidx1, midx1, mapping1 = sh_indices(lmax1)
    lidx2, midx2, mapping2 = sh_indices(lmax2)
    lidx3, midx3, mapping3 = sh_indices(lmax3)

    cg = torch.zeros((lidx1.shape[0], lidx2.shape[0], lidx3.shape[0]), dtype=complex_dtype)
    for c1, (l1, m1) in enumerate(zip(lidx1.numpy(), midx1.numpy())):
        for c2, (l2, m2) in enumerate(zip(lidx2.numpy(), midx2.numpy())):
            for c3, (l3, m3) in enumerate(zip(lidx3.numpy(), midx3.numpy())):
                if abs(l1 - l2) <= l3 <= min(l1 + l2, l3) and m3 in {
                    m1 + m2,
                    m1 - m2,
                    m2 - m1,
                    -m1 - m2,
                }:
                    cg[c1, c2, c3] = float(
                        clebsch_gordan(int(l1), int(l2), int(l3), int(m1), int(m2), int(m3))
                    )

    complex_to_real1 = generate_sh_to_rsh(lmax1).to(cg.dtype)
    complex_to_real2 = generate_sh_to_rsh(lmax2).to(cg.dtype)
    complex_to_real3 = generate_sh_to_rsh(lmax3).to(cg.dtype)
    cg_rsh = torch.einsum(
        "ijk,mi,nj,ok->mno",
        cg,
        complex_to_real1,
        complex_to_real2,
        complex_to_real3.conj(),
    )
    cg_sparse = []
    new_instructions = []
    for l1, p1, l2, p2, l3, p3 in instructions:
        # note, this normalization is not used in standard schnetpack
        normalization = math.sqrt(1.0 / sum(1 for i in instructions if (i[4], i[5]) == (l3, p3)))

        count = 0
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                for m3 in range(-l3, l3 + 1):
                    dim1 = mapping1[(l1, m1)]
                    dim2 = mapping2[(l2, m2)]
                    dim3 = mapping3[(l3, m3)]
                    cg_value = (
                        cg_rsh[dim1, dim2, dim3] * (1.0j) ** (l1 + l2 - l3)
                    ).real * normalization
                    if cg_value != 0:
                        count += 1
                        cg_sparse.append(cg_value)
                        new_instructions.append((l1, m1, p1, l2, m2, p2, l3, m3, p3))

    cg_sparse_tensor = torch.tensor(cg_sparse, dtype=dtype, requires_grad=False)
    logging.debug(
        f"Dense {cg_rsh.numel():8d} Sparse {torch.sum(cg_rsh != 0):8d} Instruction {len(cg_sparse):8d}"
    )
    return cg_sparse_tensor, new_instructions


@lru_cache(maxsize=10)
def generate_sh_to_rsh(lmax: int) -> torch.Tensor:
    """Generate transformation matrix to convert (complex) spherical harmonics to real form"""
    lidx, midx, _ = sh_indices(lmax)

    l1: torch.Tensor = lidx[:, None]
    l2: torch.Tensor = lidx[None, :]
    m1: torch.Tensor = midx[:, None]
    m2: torch.Tensor = midx[None, :]
    U: torch.Tensor = (
        1.0 * ((m1 == 0) * (m2 == 0))
        + (-1.0) ** abs(m1) / math.sqrt(2) * ((m1 == m2) * (m1 > 0))
        + 1.0 / math.sqrt(2) * ((m1 == -m2) * (m2 < 0))
        + -1.0j * (-1.0) ** abs(m1) / math.sqrt(2) * ((m1 == -m2) * (m1 < 0))
        + 1.0j / math.sqrt(2) * ((m1 == m2) * (m1 < 0))
    ) * (l1 == l2)
    return U
