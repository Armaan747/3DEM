"""Trilinear hexahedral (hex8) elements on a regular voxel grid.

All elements are identical cubes, so the stiffness of every element is a
linear combination of five precomputed unit-cube matrices (one per material
harmonic, see :mod:`threedem.materials`):

    Ke = size * fill**p * sum_m a_m * K_m

with ``a = (1, <cos2t>, <sin2t>, <cos4t>, <sin4t>)`` taken from the voxelizer.
"""

from __future__ import annotations

import numpy as np

# Corner order of a voxel (i, j, k): offsets of its eight nodes.
CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)
_SIGNS = 2.0 * CORNERS - 1.0


def _b_matrix(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Strain-displacement matrix (6x24) of the unit cube at natural coordinates."""
    s = _SIGNS
    dn_dxi = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dn_deta = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta)
    dn_dzeta = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
    # Unit cube: x = (xi + 1) / 2, so d/dx = 2 d/dxi.
    gx, gy, gz = 2 * dn_dxi, 2 * dn_deta, 2 * dn_dzeta
    B = np.zeros((6, 24))
    B[0, 0::3] = gx
    B[1, 1::3] = gy
    B[2, 2::3] = gz
    B[3, 1::3], B[3, 2::3] = gz, gy
    B[4, 0::3], B[4, 2::3] = gz, gx
    B[5, 0::3], B[5, 1::3] = gy, gx
    return B


def center_b_matrix(size: float) -> np.ndarray:
    """Strain-displacement matrix at the element centre for a cube of edge ``size``."""
    return _b_matrix(0.0, 0.0, 0.0) / size


def unit_stiffness(basis: np.ndarray) -> np.ndarray:
    """``(5, 24, 24)`` unit-cube stiffness for each of the five material basis matrices."""
    g = 1 / np.sqrt(3)
    det_j = 1 / 8
    out = np.zeros((basis.shape[0], 24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _b_matrix(xi, eta, zeta)
                for m in range(basis.shape[0]):
                    out[m] += B.T @ basis[m] @ B * det_j
    return out


def element_nodes(ijk: np.ndarray, node_shape: tuple[int, int, int]) -> np.ndarray:
    """Flat full-grid node ids ``(n, 8)`` for voxel indices ``ijk`` of shape ``(n, 3)``."""
    ny1, nz1 = node_shape[1], node_shape[2]
    corner = ijk[:, None, :] + CORNERS[None, :, :]
    return (corner[..., 0] * ny1 + corner[..., 1]) * nz1 + corner[..., 2]
