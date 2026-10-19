"""Static linear solver: boundary conditions, assembly, solution and stress recovery."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .fe import center_b_matrix, element_nodes, unit_stiffness
from .materials import Material, get_material
from .voxelize import VoxelModel

_CHUNK = 20000
_AXES = {"x": 0, "y": 1, "z": 2}
_FACES = {"xmin": (0, -1), "xmax": (0, 1), "ymin": (1, -1), "ymax": (1, 1), "zmin": (2, -1), "zmax": (2, 1)}


@dataclass
class Result:
    """Solution on the voxel grid.  Cell fields are ``nan`` where there is no material."""

    displacement: np.ndarray  # (nx+1, ny+1, nz+1, 3) mm, zero at unused nodes
    cell: dict[str, np.ndarray]  # name -> (nx, ny, nz)
    reaction: np.ndarray  # (3,) total reaction force at fixed nodes, N
    applied: np.ndarray  # (3,) total applied force, N
    info: dict = field(default_factory=dict)

    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.displacement, axis=-1).max())

    def summary(self) -> str:
        vm = self.cell["von_mises"]
        sf = self.cell["safety_factor"]
        return "\n".join(
            [
                f"elements: {self.info['n_elements']}   dofs: {self.info['n_dofs']}   solver: {self.info['solver']}",
                f"max displacement: {self.max_displacement():.4g} mm",
                f"max von Mises (material): {np.nanmax(vm):.4g} MPa",
                f"min safety factor (preliminary): {np.nanmin(sf):.3g}",
                f"applied force: {np.array2string(self.applied, precision=4)} N   "
                f"reaction: {np.array2string(self.reaction, precision=4)} N",
            ]
        )


class Model:
    """A voxel model with a material, supports and loads."""

    def __init__(self, voxels: VoxelModel, material="PLA", min_fill: float = 0.05, fill_exponent: float = 1.0):
        self.voxels = voxels
        self.material: Material = get_material(material)
        self.fill_exponent = fill_exponent
        nx, ny, nz = voxels.shape
        self.node_shape = (nx + 1, ny + 1, nz + 1)

        ijk = np.argwhere(voxels.fill >= min_fill)
        if len(ijk) == 0:
            raise ValueError("the voxel model has no elements above min_fill")
        self._ijk = ijk
        self._fill = voxels.fill[tuple(ijk.T)].astype(float)
        self._harm = np.concatenate(
            [np.ones((len(ijk), 1)), voxels.orient[tuple(ijk.T)].astype(float)], axis=1
        )  # (n, 5): 1, c2, s2, c4, s4

        flat = element_nodes(ijk, self.node_shape)
        self._node_flat, inverse = np.unique(flat, return_inverse=True)
        self._conn = inverse.reshape(-1, 8)  # element -> compact node index
        self.n_nodes = len(self._node_flat)
        gi = np.stack(np.unravel_index(self._node_flat, self.node_shape), axis=1)
        self.node_xyz = voxels.origin[None, :] + gi * voxels.size

        self._fixed: dict[int, float] = {}  # dof -> prescribed value
        self._force = np.zeros(3 * self.n_nodes)

    # -- public views of the mesh (used by the GUI) ------------------------------------

    @property
    def elements_ijk(self) -> np.ndarray:
        """``(n, 3)`` voxel indices of the elements."""
        return self._ijk

    @property
    def node_grid_index(self) -> np.ndarray:
        """``(n_nodes, 3)`` full-grid index of every node."""
        return np.stack(np.unravel_index(self._node_flat, self.node_shape), axis=1)

    def node_ids_from_grid(self, flat_grid_ids: np.ndarray) -> np.ndarray:
        """Map flat full-grid node ids (that belong to the mesh) to compact node indices."""
        return np.searchsorted(self._node_flat, flat_grid_ids)

    def clear_bcs(self) -> None:
        """Remove all supports and loads."""
        self._fixed.clear()
        self._force[:] = 0.0

    # -- selection -------------------------------------------------------------------

    def select_nodes(self, box=None, face: str | None = None, depth: float = 0.0) -> np.ndarray:
        """Indices of nodes inside ``box`` or on a model ``face``.

        ``box`` is ``{"x": [lo, hi], "y": [...], "z": [...]}`` in G-code coordinates
        (mm); ``None`` bounds are open.  ``face`` is one of xmin, xmax, ymin, ymax,
        zmin, zmax and selects nodes within ``depth`` mm of the model's extreme.
        """
        tol = 1e-6 * self.voxels.size + 1e-9
        mask = np.ones(self.n_nodes, dtype=bool)
        if face is not None:
            try:
                axis, sign = _FACES[face.lower()]
            except KeyError:
                raise ValueError(f"unknown face {face!r}; use one of {sorted(_FACES)}") from None
            c = self.node_xyz[:, axis]
            if sign < 0:
                mask &= c <= c.min() + depth + tol
            else:
                mask &= c >= c.max() - depth - tol
        if box is not None:
            for name, rng in box.items():
                axis = _AXES[name.lower()]
                lo, hi = rng
                c = self.node_xyz[:, axis]
                if lo is not None:
                    mask &= c >= lo - tol
                if hi is not None:
                    mask &= c <= hi + tol
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"selection box={box} face={face} matched no nodes")
        return idx

    # -- boundary conditions ---------------------------------------------------------

    def fix(self, nodes, axes: str = "xyz", value: float = 0.0) -> None:
        """Constrain the given displacement components of ``nodes`` to ``value`` (mm)."""
        for ax in axes.lower():
            for n in np.asarray(nodes):
                self._fixed[3 * int(n) + _AXES[ax]] = float(value)

    def add_force(self, nodes, force) -> None:
        """Apply a total force vector (N) spread over ``nodes`` by tributary area.

        Nodes belonging to full grid-aligned square patches get area-consistent
        weights, so a uniform traction is reproduced; otherwise it is shared equally.
        """
        nodes = np.asarray(nodes)
        w = self._tributary_weights(nodes)
        force = np.asarray(force, dtype=float)
        for a in range(3):
            self._force[3 * nodes + a] += force[a] * w

    def _tributary_weights(self, nodes) -> np.ndarray:
        sel = np.zeros(self.node_shape, dtype=bool)
        gi = np.stack(np.unravel_index(self._node_flat[nodes], self.node_shape), axis=1)
        sel[tuple(gi.T)] = True
        w = np.zeros(self.node_shape)
        for ax in range(3):
            # unit quads lying in the plane normal to `ax` whose four corners are all selected
            a, b = [i for i in range(3) if i != ax]
            na, nb = self.node_shape[a] - 1, self.node_shape[b] - 1
            corners = []
            for da in (0, 1):
                for db in (0, 1):
                    s2 = [slice(None)] * 3
                    s2[a], s2[b] = slice(da, na + da), slice(db, nb + db)
                    corners.append(tuple(s2))
            q = np.logical_and.reduce([sel[c] for c in corners])
            for c in corners:
                w[c] += 0.25 * q
        weights = w[tuple(gi.T)]
        if weights.sum() <= 0:
            weights = np.ones(len(nodes))
        return weights / weights.sum()

    # -- solution --------------------------------------------------------------------

    def solve(self, solver: str = "auto", tol: float = 1e-9, direct_limit: int = 60000) -> Result:
        vox = self.voxels
        size = vox.size
        mat = self.material
        basis = mat.harmonic_basis()  # (5, 6, 6)
        k_unit = unit_stiffness(basis) * size  # (5, 24, 24)
        scale = self._fill**self.fill_exponent

        if not self._fixed:
            raise ValueError("no supports defined: the model would be free to move")

        # Drop islands that are not connected to any support: they would be singular.
        keep_el, n_pruned = self._prune_floating()
        conn = self._conn[keep_el]
        harm = self._harm[keep_el]
        scale_k = scale[keep_el]
        active = np.zeros(self.n_nodes, dtype=bool)
        active[conn.ravel()] = True

        fixed_dofs = np.array(sorted(self._fixed), dtype=np.int64)
        fixed_vals = np.array([self._fixed[d] for d in fixed_dofs])
        if not active[fixed_dofs // 3].all():
            warnings.warn("some fixed nodes are not part of the connected model")
        load_nodes = np.flatnonzero(np.abs(self._force.reshape(-1, 3)).sum(axis=1) > 0)
        if not active[load_nodes].all():
            warnings.warn("some loaded nodes are disconnected from every support and carry no load")

        n_dof = 3 * self.n_nodes
        K = self._assemble(conn, harm, scale_k, k_unit, n_dof)

        is_active_dof = np.repeat(active, 3)
        is_fixed = np.zeros(n_dof, dtype=bool)
        is_fixed[fixed_dofs] = True
        free = np.flatnonzero(is_active_dof & ~is_fixed)
        fixed_active = fixed_dofs[is_active_dof[fixed_dofs]]

        u = np.zeros(n_dof)
        u[fixed_dofs] = fixed_vals
        f = np.where(is_active_dof, self._force, 0.0)
        rhs = f[free] - K[free][:, fixed_active] @ u[fixed_active]
        Kff = K[free][:, free].tocsc()

        used = solver
        if solver == "auto":
            used = "direct" if len(free) <= direct_limit else "cg"
        if used == "direct":
            u[free] = spla.spsolve(Kff, rhs)
        elif used == "cg":
            u[free], info = _cg(Kff, rhs, tol)
            if info != 0:
                warnings.warn(f"CG did not converge (info={info}); results may be inaccurate")
        else:
            raise ValueError(f"unknown solver {solver!r}")

        reaction_all = K @ u - f
        reaction = np.bincount(fixed_active % 3, weights=reaction_all[fixed_active], minlength=3)
        applied = f.reshape(-1, 3).sum(axis=0)

        cell = self._recover(u, keep_el, conn, harm, basis, mat)
        disp = np.zeros((*self.node_shape, 3))
        gi = np.stack(np.unravel_index(self._node_flat, self.node_shape), axis=1)
        disp[tuple(gi.T)] = u.reshape(-1, 3)
        info = {
            "n_elements": int(len(conn)),
            "n_dofs": int(len(free)),
            "solver": used,
            "floating_elements_removed": int(n_pruned),
        }
        if n_pruned:
            warnings.warn(f"{n_pruned} elements not connected to any support were removed")
        return Result(disp, cell, reaction, applied, info)

    def _prune_floating(self):
        n_el = len(self._conn)
        rows = np.repeat(self._conn[:, 0], 7)
        cols = self._conn[:, 1:].ravel()
        g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        _, label = connected_components(g, directed=False)
        fixed_nodes = np.unique(np.array(sorted(self._fixed), dtype=np.int64) // 3)
        supported = np.unique(label[fixed_nodes])
        keep = np.isin(label[self._conn[:, 0]], supported)
        return keep, int(n_el - keep.sum())

    def _assemble(self, conn, harm, scale, k_unit, n_dof):
        K = sp.csr_matrix((n_dof, n_dof))
        for s in range(0, len(conn), _CHUNK):
            e = slice(s, s + _CHUNK)
            ke = np.einsum("em,mij->eij", harm[e], k_unit) * scale[e, None, None]
            dofs = (3 * conn[e][:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
            rows = np.repeat(dofs, 24, axis=1).ravel()
            cols = np.tile(dofs, (1, 24)).ravel()
            K = K + sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        return K

    def _recover(self, u, keep_el, conn, harm, basis, mat: Material):
        size = self.voxels.size
        Bc = center_b_matrix(size)
        dofs = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
        ue = u[dofs]
        eps = ue @ Bc.T  # (n, 6)
        c_eff = np.einsum("em,mij->eij", harm, basis)
        sig = np.einsum("eij,ej->ei", c_eff, eps)  # stress in the deposited material, MPa
        sxx, syy, szz, tyz, txz, txy = sig.T
        vm = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (tyz**2 + txz**2 + txy**2)
        )
        in_plane = np.sqrt(np.maximum(sxx**2 - sxx * syy + syy**2 + 3 * txy**2, 0))
        fi_xy = in_plane / min(mat.strength_1, mat.strength_2)
        fi_z = np.sqrt(
            (np.maximum(szz, 0) / mat.strength_3) ** 2 + (txz**2 + tyz**2) / mat.shear_strength_3**2
        )
        fi = np.maximum(fi_xy, fi_z)
        sf = np.minimum(1.0 / np.maximum(fi, 1e-12), 1e3)

        shape = self.voxels.shape
        sel = tuple(self._ijk[keep_el].T)

        def grid(values):
            a = np.full(shape, np.nan, dtype=np.float32)
            a[sel] = values
            return a

        return {
            "fill": grid(self._fill[keep_el]),
            "von_mises": grid(vm),
            "sigma_xx": grid(sxx), "sigma_yy": grid(syy), "sigma_zz": grid(szz),
            "tau_yz": grid(tyz), "tau_xz": grid(txz), "tau_xy": grid(txy),
            "index_in_plane": grid(fi_xy),
            "index_interlayer": grid(fi_z),
            "safety_factor": grid(sf),
        }


def _cg(A, b, tol):
    d = A.diagonal()
    M = sp.diags(1.0 / np.where(d > 0, d, 1.0))
    try:
        import pyamg  # optional accelerator

        ml = pyamg.smoothed_aggregation_solver(A)
        M = ml.aspreconditioner()
    except ImportError:
        pass
    x, info = spla.cg(A, b, rtol=tol, atol=0.0, M=M, maxiter=20000)
    return x, info
