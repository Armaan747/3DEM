"""Orthotropic material model for extruded (FDM) material.

Local axes of a deposited line:

* 1 - along the line (raster direction)
* 2 - across the line, in the build plane (road-to-road bond)
* 3 - build direction Z (layer-to-layer bond)

Voigt order is ``[xx, yy, zz, yz, xz, xy]`` with engineering shear strains.
Units are N, mm, MPa throughout.

The preset values are *illustrative* typical numbers, not measurements. Print and
break coupons of your own material and printer and override them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Rotation about Z only, so C(theta) is a trigonometric polynomial of degree 4
# in theta.  It therefore lies in the span of these five functions, which lets
# the voxelizer store just five accumulated numbers per voxel instead of a 6x6.
N_HARMONICS = 5


def strain_transform(theta: float) -> np.ndarray:
    """6x6 matrix mapping global engineering strain to strain in axes rotated by theta about Z."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c * c, s * s, 0, 0, 0, c * s],
            [s * s, c * c, 0, 0, 0, -c * s],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [-2 * c * s, 2 * c * s, 0, 0, 0, c * c - s * s],
        ]
    )


def stress_transform(theta: float) -> np.ndarray:
    """6x6 matrix mapping global stress to stress in axes rotated by theta about Z."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c * c, s * s, 0, 0, 0, 2 * c * s],
            [s * s, c * c, 0, 0, 0, -2 * c * s],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [-c * s, c * s, 0, 0, 0, c * c - s * s],
        ]
    )


@dataclass(frozen=True)
class Material:
    """Orthotropic elastic constants and strengths of printed material (MPa)."""

    name: str
    E1: float  # along the raster line
    E2: float  # across the line, in plane
    E3: float  # build direction (layer bond)
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float
    strength_1: float = 50.0  # tensile strength along the line
    strength_2: float = 40.0  # tensile strength across the line
    strength_3: float = 25.0  # tensile strength across layers
    shear_strength_3: float = 20.0  # interlayer shear strength

    def local_stiffness(self) -> np.ndarray:
        """6x6 stiffness in the local line axes."""
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / self.E1, 1 / self.E2, 1 / self.E3
        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / self.G23, 1 / self.G13, 1 / self.G12
        C = np.linalg.inv(S)
        if np.any(np.linalg.eigvalsh(0.5 * (C + C.T)) <= 0):
            raise ValueError(f"material {self.name!r} is not positive definite")
        return C

    def stiffness(self, theta: float) -> np.ndarray:
        """6x6 stiffness with the line direction rotated by theta about Z (radians from +X)."""
        T = strain_transform(theta)
        return T.T @ self.local_stiffness() @ T

    def harmonic_basis(self) -> np.ndarray:
        """Return ``(5, 6, 6)`` matrices ``B_m`` with ``C(theta) = sum_m f_m(theta) B_m``.

        ``f = (1, cos 2t, sin 2t, cos 4t, sin 4t)``.
        """
        thetas = np.linspace(0, np.pi, 16, endpoint=False)
        F = np.stack(harmonics(thetas), axis=1)  # (16, 5)
        Cs = np.stack([self.stiffness(t).ravel() for t in thetas])  # (16, 36)
        coef, *_ = np.linalg.lstsq(F, Cs, rcond=None)
        return coef.reshape(N_HARMONICS, 6, 6)

    def with_(self, **kwargs) -> "Material":
        return replace(self, **kwargs)


def harmonics(theta):
    """The five basis functions evaluated at theta (list of arrays)."""
    theta = np.asarray(theta, dtype=float)
    return [
        np.ones_like(theta),
        np.cos(2 * theta),
        np.sin(2 * theta),
        np.cos(4 * theta),
        np.sin(4 * theta),
    ]


# Illustrative values only - see module docstring.
PRESETS: dict[str, Material] = {
    "PLA": Material("PLA", 3400, 3000, 2600, 1200, 1000, 1000, 0.35, 0.35, 0.35, 55, 45, 28, 22),
    "PETG": Material("PETG", 2100, 1900, 1600, 750, 650, 650, 0.38, 0.38, 0.38, 45, 38, 30, 24),
    "ABS": Material("ABS", 2100, 1900, 1500, 750, 650, 650, 0.38, 0.38, 0.38, 35, 30, 20, 16),
    "ISOTROPIC": Material("ISOTROPIC", 3000, 3000, 3000, 1111.11, 1111.11, 1111.11, 0.35, 0.35, 0.35, 50, 50, 50, 30),
}


def get_material(spec) -> Material:
    """Resolve a preset name, a ``Material`` or a dict of fields (optionally with ``base``)."""
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        try:
            return PRESETS[spec.upper()]
        except KeyError:
            raise ValueError(f"unknown material {spec!r}; presets: {sorted(PRESETS)}") from None
    if isinstance(spec, dict):
        spec = dict(spec)
        base = spec.pop("base", None)
        if base is not None:
            return get_material(base).with_(**spec)
        return Material(**spec)
    raise TypeError(f"cannot build a material from {type(spec).__name__}")
