"""Convert toolpaths into a voxel grid of fill fraction and raster orientation.

Every extruded segment deposits its (E-derived) plastic volume onto the voxels
it passes through, so the total plastic volume is conserved.  Besides volume,
each voxel accumulates the volume-weighted harmonics ``cos 2t, sin 2t, cos 4t,
sin 4t`` of the raster direction ``t``.  These are exactly what is needed to
average the rotated orthotropic stiffness of all lines inside the voxel (see
:meth:`threedem.materials.Material.harmonic_basis`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gcode import ToolPaths

_MAX_SAMPLES_PER_CHUNK = 500_000  # each sample is split over up to 8 voxels


@dataclass
class VoxelModel:
    origin: np.ndarray  # (3,) corner of voxel (0, 0, 0), mm
    size: float  # voxel edge length, mm
    fill: np.ndarray  # (nx, ny, nz) plastic volume fraction, 0..1
    orient: np.ndarray  # (nx, ny, nz, 4) volume-averaged cos2t, sin2t, cos4t, sin4t
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fill.shape

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(self.origin[a] + (np.arange(self.shape[a]) + 0.5) * self.size for a in range(3))

    def __repr__(self) -> str:
        return (
            f"VoxelModel(shape={self.shape}, size={self.size}, "
            f"filled={int((self.fill > 0).sum())}, plastic={self.fill.sum() * self.size**3:.1f} mm^3)"
        )


def voxelize(
    paths: ToolPaths,
    size: float,
    origin: tuple[float, float, float] | None = None,
    shape: tuple[int, int, int] | None = None,
) -> VoxelModel:
    """Voxelize toolpaths with cubic voxels of edge ``size`` (mm).

    A voxel edge close to the road width (about 0.4-0.5 mm) resolves individual
    walls and infill lines; larger voxels give a homogenized, cheaper model.
    ``origin``/``shape`` fix the grid explicitly (useful for exact comparisons).
    """
    if len(paths) == 0:
        raise ValueError("no extruded segments to voxelize (everything excluded?)")
    if size <= 0:
        raise ValueError("voxel size must be positive")

    if origin is None:
        lo, hi = paths.bounds()
        half_w = float(paths.width.max()) / 2
        lo = lo - np.array([half_w, half_w, 0.0])
        hi = hi + np.array([half_w, half_w, 0.0])
        lo[2] = float((np.minimum(paths.p0[:, 2], paths.p1[:, 2]) - paths.height).min())
        origin_a = lo
    else:
        origin_a = np.asarray(origin, dtype=float)
        hi = None
    if shape is None:
        if hi is None:
            lo2, hi = paths.bounds()
            hi = hi + float(paths.width.max()) / 2
        shape = tuple(int(max(1, np.ceil((hi[a] - origin_a[a]) / size - 1e-9))) for a in range(3))
    nx, ny, nz = shape
    n_vox = nx * ny * nz
    if n_vox > 400_000_000:
        raise MemoryError(f"voxel grid {shape} is too large; increase the voxel size")

    acc = np.zeros((5, n_vox))  # volume, then the four harmonic sums
    step = size / 2

    d = paths.p1 - paths.p0
    seg_len = np.linalg.norm(d, axis=1)
    # Spread each bead as a rectangle with the same cross-section area, so the
    # deposited volume is exactly the extruded volume and nothing spills past the part.
    eq_width = paths.volume / np.maximum(seg_len, 1e-12) / paths.height
    a = (paths.p0, paths.p1, paths.volume, eq_width, paths.height)
    n_al = np.maximum(1, np.ceil(seg_len / step)).astype(np.int64)
    n_wd = np.maximum(1, np.ceil(eq_width / step)).astype(np.int64)
    n_ht = np.maximum(1, np.ceil(paths.height / step)).astype(np.int64)
    cnt = n_al * n_wd * n_ht

    start = 0
    n_seg = len(paths)
    cum = np.cumsum(cnt)
    while start < n_seg:
        base = cum[start - 1] if start else 0
        stop = int(np.searchsorted(cum, base + _MAX_SAMPLES_PER_CHUNK, side="right"))
        stop = max(stop, start + 1)
        stop = min(stop, n_seg)
        _deposit(acc, a, d, seg_len, n_al, n_wd, n_ht, cnt, start, stop, origin_a, size, shape)
        start = stop

    vol = acc[0]
    fill = vol / size**3
    with np.errstate(invalid="ignore", divide="ignore"):
        orient = np.where(vol[None, :] > 0, acc[1:] / vol[None, :], 0.0)
    overfill = float((fill > 1.0).sum()) / max(1, int((fill > 0).sum()))
    fill = np.minimum(fill, 1.0)
    return VoxelModel(
        origin=origin_a,
        size=float(size),
        fill=fill.reshape(shape).astype(np.float32),
        orient=orient.T.reshape(*shape, 4).astype(np.float32),
        meta={"overfilled_fraction": overfill, "plastic_in_grid": float(vol.sum()), "plastic_total": float(paths.volume.sum())},
    )


def _split_1d(lo, hi):
    """Split the interval [lo, hi] (width <= 1 cell) over cells i and i+1: (i, weight_i, weight_i+1)."""
    i = np.floor(lo).astype(np.int64)
    w0 = np.clip((i + 1 - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
    return i, w0, 1.0 - w0


def _deposit(acc, a, d, seg_len, n_al, n_wd, n_ht, cnt, i0, i1, origin, size, shape):
    """Deposit segments ``i0:i1``.

    Each segment is cut into small cells (along x across x vertical).  Every cell
    carries an equal share of the segment volume and is split *exactly* among the
    voxels it overlaps, so plastic is conserved and there is no aliasing against
    the grid for axis-aligned roads.  The road is modelled as a rectangle of the
    same cross-section area as the extruded bead (``width`` is already that
    equal-area width).
    """
    p0, _p1, volume, width, height = a
    c = cnt[i0:i1]
    total = int(c.sum())
    seg = np.repeat(np.arange(i0, i1), c)
    first = np.cumsum(c) - c
    local = np.arange(total) - np.repeat(first, c)

    nw = n_wd[seg]
    nh = n_ht[seg]
    per_al = nw * nh
    i_al = local // per_al
    rem = local % per_al
    i_wd = rem // nh
    i_ht = rem % nh

    L = np.maximum(seg_len[seg], 1e-12)
    d_al = L / n_al[seg]  # cell extent along the road
    d_wd = width[seg] / nw  # across the road
    d_ht = height[seg] / nh  # vertical
    t = (i_al + 0.5) / n_al[seg]
    u = ((i_wd + 0.5) / nw - 0.5) * width[seg]
    dz = ((i_ht + 0.5) / nh - 0.5) * height[seg]

    dirv = d[seg]
    ux, uy = dirv[:, 0] / L, dirv[:, 1] / L
    pos = p0[seg] + dirv * t[:, None]
    px = pos[:, 0] - u * uy  # across-line offset uses the in-plane normal (-uy, ux)
    py = pos[:, 1] + u * ux
    pz = pos[:, 2] - 0.5 * height[seg] + dz  # G-code Z is the top of the layer

    # axis-aligned footprint of the (possibly rotated) cell
    ex = np.abs(ux) * d_al + np.abs(uy) * d_wd
    ey = np.abs(uy) * d_al + np.abs(ux) * d_wd

    cx, cy, cz = (px - origin[0]) / size, (py - origin[1]) / size, (pz - origin[2]) / size
    hx, hy, hz = 0.5 * ex / size, 0.5 * ey / size, 0.5 * d_ht / size
    ix, wx0, wx1 = _split_1d(cx - hx, cx + hx)
    iy, wy0, wy1 = _split_1d(cy - hy, cy + hy)
    iz, wz0, wz1 = _split_1d(cz - hz, cz + hz)

    w = volume[seg] / cnt[seg]
    theta = np.arctan2(uy, ux)
    c2, s2, c4, s4 = np.cos(2 * theta), np.sin(2 * theta), np.cos(4 * theta), np.sin(4 * theta)

    flats, weights = [], []
    for ox, wx in ((0, wx0), (1, wx1)):
        for oy, wy in ((0, wy0), (1, wy1)):
            for oz, wz in ((0, wz0), (1, wz1)):
                jx, jy, jz = ix + ox, iy + oy, iz + oz
                wt = w * wx * wy * wz
                ok = (
                    (wt > 0) & (jx >= 0) & (jx < shape[0]) & (jy >= 0) & (jy < shape[1])
                    & (jz >= 0) & (jz < shape[2])
                )
                flats.append(((jx * shape[1] + jy) * shape[2] + jz)[ok])
                weights.append((wt[ok], c2[ok], s2[ok], c4[ok], s4[ok]))
    flat = np.concatenate(flats)
    wt = np.concatenate([x[0] for x in weights])
    n_vox = acc.shape[1]
    acc[0] += np.bincount(flat, wt, minlength=n_vox)
    for k in range(1, 5):
        h = np.concatenate([x[k] for x in weights])
        acc[k] += np.bincount(flat, wt * h, minlength=n_vox)
