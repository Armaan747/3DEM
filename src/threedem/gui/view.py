"""3D scene on top of a PyVista plotter: toolpaths, voxel surface, selection, supports, loads."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

from ..gcode import ToolPaths
from ..solve import Model
from .geometry import QuadSet

# distinguishable colours for feature types (RGB 0-255)
PALETTE = np.array(
    [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
        (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    ],
    dtype=np.uint8,
)
SELECT_COLOR = "#ffd400"
SUPPORT_COLOR = "#1f6feb"
LOAD_COLOR = "#e5322d"


def type_color(i: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in PALETTE[i % len(PALETTE)])


def quad_polydata(points: np.ndarray, quads: np.ndarray) -> pv.PolyData:
    n = len(quads)
    faces = np.hstack([np.full((n, 1), 4, dtype=np.int64), quads.astype(np.int64)]).ravel()
    return pv.PolyData(np.asarray(points, dtype=float), faces)


class SceneView:
    def __init__(self, plotter: pv.Plotter):
        self.pl = plotter
        self.pl.set_background("#f4f5f7", top="#cfd6df")
        self.pl.add_axes()
        self._model_actor = None
        self._press_pos = None
        self._on_pick = None
        self._bounds_shown = True

    # -- toolpaths -----------------------------------------------------------------------

    def show_toolpaths(self, paths: ToolPaths, visible_types, max_layer: int | None = None,
                       color_by: str = "type", reset: bool = False) -> int:
        """Draw segments of the given feature types up to ``max_layer``.  Returns segments drawn."""
        self.clear_model()
        keep = np.isin(paths.type_id, list(visible_types))
        if max_layer is not None:
            keep &= paths.layer <= max_layer
        n = int(keep.sum())
        self.pl.remove_actor("paths")
        if n == 0:
            self.pl.render()
            return 0
        pts = np.concatenate([paths.p0[keep], paths.p1[keep]])
        idx = np.arange(n)
        lines = np.column_stack([np.full(n, 2), idx, idx + n]).ravel()
        mesh = pv.PolyData(pts, lines=lines)
        if color_by == "layer":
            from matplotlib import colormaps

            lay = paths.layer[keep].astype(float)
            span = max(1.0, float(paths.layer.max()))
            rgb = (colormaps["viridis"](lay / span)[:, :3] * 255).astype(np.uint8)
        else:
            rgb = PALETTE[paths.type_id[keep] % len(PALETTE)]
        mesh.cell_data["rgb"] = rgb
        self.pl.add_mesh(mesh, scalars="rgb", rgb=True, line_width=1.5, name="paths", reset_camera=False,
                         pickable=False)
        self._after_scene(mesh.bounds, reset)
        return n

    # -- voxel surface -------------------------------------------------------------------

    def clear_model(self) -> None:
        for name in ("model", "model_ghost", "sel", "supports", "loads", "arrows"):
            self.pl.remove_actor(name)
        self._remove_scalar_bars()
        self._model_actor = None

    def hide_toolpaths(self) -> None:
        self.pl.remove_actor("paths")

    def show_model(self, model: Model, quads: QuadSet, cell_values=None, point_values=None, title: str = "",
                   cmap: str = "viridis", clim=None, points=None, edges: bool = False, ghost: bool = False,
                   reset: bool = False) -> None:
        """Draw the exposed faces; colour by per-quad ``cell_values`` or per-node ``point_values``."""
        self.hide_toolpaths()
        self._remove_scalar_bars()
        base_pts = model.node_xyz
        mesh = quad_polydata(base_pts if points is None else points, quads.nodes)
        kwargs = dict(name="model", reset_camera=False, pickable=True, show_edges=edges,
                      edge_color="#555555", smooth_shading=False)
        if cell_values is not None:
            mesh.cell_data["v"] = np.asarray(cell_values, dtype=float)
            kwargs.update(scalars="v", cmap=cmap, clim=clim, nan_color="#bbbbbb")
        elif point_values is not None:
            mesh.point_data["v"] = np.asarray(point_values, dtype=float)
            kwargs.update(scalars="v", cmap=cmap, clim=clim, nan_color="#bbbbbb")
        else:
            kwargs.update(color="#c9ced6")
        if cell_values is not None or point_values is not None:
            kwargs["scalar_bar_args"] = dict(title=title.replace(" (", "\n("), vertical=True, position_x=0.88,
                                             position_y=0.12, height=0.7, width=0.04, n_labels=6, fmt="%.3g",
                                             title_font_size=12, label_font_size=11)
        self._model_actor = self.pl.add_mesh(mesh, **kwargs)
        self.pl.remove_actor("model_ghost")
        if ghost and points is not None:
            self.pl.add_mesh(quad_polydata(base_pts, quads.nodes), style="wireframe", color="#888888",
                             opacity=0.25, name="model_ghost", pickable=False, reset_camera=False)
        self._after_scene(mesh.bounds, reset)

    def _remove_scalar_bars(self) -> None:
        for t in list(self.pl.scalar_bars.keys()):
            self.pl.remove_scalar_bar(t)

    # -- overlays ------------------------------------------------------------------------

    def _offset_quads(self, model: Model, quads: QuadSet, idx: np.ndarray, points=None) -> pv.PolyData:
        pts = (model.node_xyz if points is None else points)[quads.nodes[idx]]  # (n, 4, 3)
        eps = 0.03 * model.voxels.size
        pts = pts + quads.normals()[idx][:, None, :] * eps
        n = len(idx)
        return quad_polydata(pts.reshape(-1, 3), np.arange(4 * n).reshape(n, 4))

    def show_selection(self, model: Model, quads: QuadSet, idx: np.ndarray, points=None) -> None:
        self.pl.remove_actor("sel")
        if len(idx):
            self.pl.add_mesh(self._offset_quads(model, quads, idx, points), color=SELECT_COLOR, name="sel",
                             pickable=False, reset_camera=False, show_edges=True, edge_color="#8a6d00")
        self.pl.render()

    def show_bcs(self, model: Model, quads: QuadSet, supports, loads, points=None) -> None:
        """``supports``: list of quad-index arrays; ``loads``: list of (quad-index array, force vector)."""
        for name in ("supports", "loads", "arrows"):
            self.pl.remove_actor(name)
        sup = [s for s in supports if len(s)]
        if sup:
            m = self._offset_quads(model, quads, np.unique(np.concatenate(sup)), points)
            self.pl.add_mesh(m, color=SUPPORT_COLOR, name="supports", pickable=False, reset_camera=False)
        ld = [(i, f) for i, f in loads if len(i)]
        if ld:
            m = self._offset_quads(model, quads, np.unique(np.concatenate([i for i, _ in ld])), points)
            self.pl.add_mesh(m, color=LOAD_COLOR, name="loads", pickable=False, reset_camera=False)
            self._draw_arrows(model, quads, ld, points)
        self.pl.render()

    def _draw_arrows(self, model: Model, quads: QuadSet, loads, points) -> None:
        pts = model.node_xyz if points is None else points
        extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        length = 0.22 * extent
        starts, dirs = [], []
        for idx, force in loads:
            f = np.asarray(force, dtype=float)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            d = f / norm
            centre = pts[quads.nodes[idx]].reshape(-1, 3).mean(axis=0)
            pulling = float((quads.normals()[idx].mean(axis=0)) @ d) > 0
            starts.append(centre if pulling else centre - d * length)  # tip always touches the face
            dirs.append(d)
        if not starts:
            return
        cloud = pv.PolyData(np.array(starts))
        cloud["dir"] = np.array(dirs)
        arrows = cloud.glyph(orient="dir", scale=False, factor=length, geom=pv.Arrow(tip_length=0.35,
                                                                                    tip_radius=0.08, shaft_radius=0.03))
        self.pl.add_mesh(arrows, color=LOAD_COLOR, name="arrows", pickable=False, reset_camera=False)

    # -- camera / misc -------------------------------------------------------------------

    def _after_scene(self, bounds, reset: bool) -> None:
        if self._bounds_shown:
            self.pl.remove_bounds_axes()
            self.pl.show_bounds(bounds=bounds, grid=False, location="outer", ticks="outside",
                                xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=8,
                                n_xlabels=4, n_ylabels=4, n_zlabels=4, color="#444444")
        if reset:
            self.pl.view_isometric()
            self.pl.reset_camera()
        self.pl.render()

    def set_bounds_visible(self, visible: bool) -> None:
        self._bounds_shown = visible
        if not visible:
            self.pl.remove_bounds_axes()
        self.pl.render()

    def view(self, which: str) -> None:
        {"iso": self.pl.view_isometric, "top": self.pl.view_xy, "front": self.pl.view_xz,
         "side": self.pl.view_yz}[which]()
        self.pl.reset_camera()
        self.pl.render()

    # -- picking -------------------------------------------------------------------------

    def enable_picking(self, callback) -> None:
        """Call ``callback(quad_index, ctrl_held)`` for a left click (not drag) on the voxel surface."""
        self._on_pick = callback
        iren = self.pl.iren
        iren.add_observer("LeftButtonPressEvent", self._press)
        iren.add_observer("LeftButtonReleaseEvent", self._release)

    def _press(self, *_):
        self._press_pos = self.pl.iren.interactor.GetEventPosition()

    def _release(self, *_):
        if self._press_pos is None or self._on_pick is None:
            return
        x, y = self.pl.iren.interactor.GetEventPosition()
        px, py = self._press_pos
        self._press_pos = None
        if abs(x - px) + abs(y - py) > 4:  # it was a camera drag
            return
        cid = self.pick_at(x, y)
        if cid is not None:
            self._on_pick(cid, bool(self.pl.iren.interactor.GetControlKey()))

    def pick_at(self, x: int, y: int) -> int | None:
        if self._model_actor is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(self._model_actor)
        picker.Pick(x, y, 0, self.pl.renderer)
        cid = picker.GetCellId()
        return None if cid < 0 else int(cid)
