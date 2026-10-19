"""Surface geometry of a voxel model for display and face picking (no Qt, no VTK needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..solve import Model

AXIS_NAMES = "xyz"


@dataclass
class QuadSet:
    """Boundary faces of the solid, one quad per exposed voxel face."""

    nodes: np.ndarray  # (n, 4) compact node indices (model.node_xyz rows)
    elem: np.ndarray  # (n, 3) voxel index of the adjacent solid element
    axis: np.ndarray  # (n,) normal axis 0/1/2
    sign: np.ndarray  # (n,) +1 / -1 outward normal direction
    plane: np.ndarray  # (n,) node-plane index along `axis`
    cell: np.ndarray  # (n, 2) in-plane voxel indices (along the two other axes)

    def __len__(self) -> int:
        return len(self.axis)

    def normals(self) -> np.ndarray:
        n = np.zeros((len(self), 3))
        n[np.arange(len(self)), self.axis] = self.sign
        return n

    def key(self) -> np.ndarray:
        """Integer id of the (axis, sign, plane) a quad lies in."""
        return (self.axis * 2 + (self.sign > 0)) * 100000 + self.plane


def boundary_quads(model: Model, occupancy: np.ndarray | None = None) -> QuadSet:
    """Exposed faces of the model's elements.

    ``occupancy`` (bool grid) restricts the solid, e.g. for a section view: faces
    revealed by the cut become boundary faces automatically.
    """
    shape = model.voxels.shape
    occ = np.zeros(shape, dtype=bool)
    occ[tuple(model.elements_ijk.T)] = True
    if occupancy is not None:
        occ &= occupancy

    ny1, nz1 = model.node_shape[1], model.node_shape[2]
    out = {k: [] for k in ("nodes", "elem", "axis", "sign", "plane", "cell")}
    for ax in range(3):
        p, q = [a for a in range(3) if a != ax]
        o = np.moveaxis(occ, ax, 0)  # (n_ax, np, nq)
        pad = np.zeros((o.shape[0] + 2, o.shape[1], o.shape[2]), dtype=bool)
        pad[1:-1] = o
        for sign in (+1, -1):
            if sign > 0:  # solid below the plane, empty above: element t-1, plane t
                m = pad[:-1] & ~pad[1:]
                t, b, c = np.nonzero(m)
                e_ax = t - 1
            else:  # empty below, solid above: element t, plane t
                m = ~pad[:-1] & pad[1:]
                t, b, c = np.nonzero(m)
                e_ax = t
            if len(t) == 0:
                continue
            corner = []
            for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                idx = [None, None, None]
                idx[ax], idx[p], idx[q] = t, b + db, c + dc
                corner.append((idx[0] * ny1 + idx[1]) * nz1 + idx[2])
            flat = np.stack(corner, axis=1)
            e = [None, None, None]
            e[ax], e[p], e[q] = e_ax, b, c
            out["nodes"].append(model.node_ids_from_grid(flat))
            out["elem"].append(np.stack(e, axis=1))
            out["axis"].append(np.full(len(t), ax))
            out["sign"].append(np.full(len(t), sign))
            out["plane"].append(t)
            out["cell"].append(np.stack([b, c], axis=1))
    if not out["axis"]:
        z4 = np.zeros((0, 4), dtype=int)
        return QuadSet(z4, np.zeros((0, 3), int), *(np.zeros(0, int) for _ in range(3)), np.zeros((0, 2), int))
    return QuadSet(*(np.concatenate(out[k]) for k in ("nodes", "elem", "axis", "sign", "plane", "cell")))


def pick_region(quads: QuadSet, picked: int, whole_plane: bool = False) -> np.ndarray:
    """Indices of the quads in the flat face region containing quad ``picked``.

    Region = same normal and plane, connected through shared edges (or the entire
    plane with ``whole_plane``).
    """
    same = np.flatnonzero(quads.key() == quads.key()[picked])
    if whole_plane or len(same) == 1:
        return same
    cells = quads.cell[same]
    lo = cells.min(axis=0)
    mask = np.zeros(tuple(cells.max(axis=0) - lo + 1), dtype=bool)
    mask[tuple((cells - lo).T)] = True
    labels, _ = ndimage.label(mask)  # 4-connectivity
    target = labels[tuple(quads.cell[picked] - lo)]
    return same[labels[tuple((cells - lo).T)] == target]


def nodes_of(quads: QuadSet, idx: np.ndarray) -> np.ndarray:
    """Unique compact node indices of the given quads."""
    return np.unique(quads.nodes[idx].ravel())


def quads_within(quads: QuadSet, node_ids: np.ndarray) -> np.ndarray:
    """Indices of quads whose four corners are all in ``node_ids``."""
    return np.flatnonzero(np.isin(quads.nodes, node_ids).all(axis=1))


def section_mask(model: Model, axis: int, position: float, keep_below: bool = True) -> np.ndarray:
    """Occupancy mask that keeps voxels on one side of the plane ``coord[axis] = position``."""
    vox = model.voxels
    centre = vox.origin[axis] + (np.arange(vox.shape[axis]) + 0.5) * vox.size
    keep = centre <= position if keep_below else centre >= position
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.broadcast_to(keep.reshape(shape), vox.shape)


def node_displacement(model: Model, displacement_grid: np.ndarray) -> np.ndarray:
    """``(n_nodes, 3)`` displacement of each mesh node from a full-grid displacement array."""
    g = model.node_grid_index
    return displacement_grid[g[:, 0], g[:, 1], g[:, 2]]
