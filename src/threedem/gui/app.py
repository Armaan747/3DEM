"""threedem main window: open G-code, set material, pick supports/loads, solve, inspect."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree

from ..gcode import DEFAULT_EXCLUDE, ToolPaths, parse_gcode
from ..materials import Material, get_material
from ..solve import Model, Result
from ..voxelize import VoxelModel, voxelize
from ..vtkio import write_vtk
from .geometry import (QuadSet, boundary_quads, node_displacement, nodes_of, pick_region, quads_within,
                       section_mask)
from .jobs import run_job
from .panels import MaterialPanel
from .view import SceneView, type_color

AXES = "xyz"
FACES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
# label, kind, key, colormap
RESULT_FIELDS = [
    ("Displacement magnitude (mm)", "point", "mag", "turbo"),
    ("Displacement X (mm)", "point", 0, "turbo"),
    ("Displacement Y (mm)", "point", 1, "turbo"),
    ("Displacement Z (mm)", "point", 2, "turbo"),
    ("von Mises stress (MPa)", "cell", "von_mises", "turbo"),
    ("Safety factor (preliminary)", "cell", "safety_factor", "RdYlGn"),
    ("In-plane failure index", "cell", "index_in_plane", "turbo"),
    ("Inter-layer failure index", "cell", "index_interlayer", "turbo"),
    ("Stress xx (MPa)", "cell", "sigma_xx", "turbo"),
    ("Stress yy (MPa)", "cell", "sigma_yy", "turbo"),
    ("Stress zz - build direction (MPa)", "cell", "sigma_zz", "turbo"),
    ("Shear yz (MPa)", "cell", "tau_yz", "turbo"),
    ("Shear xz (MPa)", "cell", "tau_xz", "turbo"),
    ("Shear xy (MPa)", "cell", "tau_xy", "turbo"),
]


@dataclass
class BC:
    kind: str  # "support" | "load"
    name: str
    nodes: np.ndarray
    axes: str = "xyz"
    force: np.ndarray | None = None

    def describe(self) -> str:
        if self.kind == "support":
            return f"{self.name}  -  fixed {self.axes.upper()}  -  {len(self.nodes)} nodes"
        f = self.force
        return f"{self.name}  -  ({f[0]:.4g}, {f[1]:.4g}, {f[2]:.4g}) N  -  {len(self.nodes)} nodes"


def _dspin(lo, hi, value, decimals=2, step=1.0, suffix=""):
    s = W.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("threedem - FEM for 3D-printed parts")
        self.resize(1450, 880)

        self.paths: ToolPaths | None = None
        self.gcode_path: Path | None = None
        self.vox: VoxelModel | None = None
        self.model: Model | None = None
        self.base_quads: QuadSet | None = None
        self.disp_quads: QuadSet | None = None
        self.result: Result | None = None
        self.disp_nodes: np.ndarray | None = None
        self.result_stale = False
        self.bcs: list[BC] = []
        self._bc_count = {"support": 0, "load": 0}
        self.sel_nodes = np.empty(0, dtype=int)
        self.sel_normal = np.zeros(3)
        self.mode = "toolpaths"
        self._points: np.ndarray | None = None
        self._section_cache: tuple | None = None
        self._pending_project: dict | None = None

        self.plotter = QtInteractor(self)
        self.view = SceneView(self.plotter)
        self.view.enable_picking(self.on_pick)

        self._build_actions()
        self.tabs = W.QTabWidget()
        self.tabs.addTab(self._build_model_tab(), "1  Model")
        self.tabs.addTab(self._build_material_tab(), "2  Material")
        self.tabs.addTab(self._build_bc_tab(), "3  Supports && Loads")
        self.tabs.addTab(self._build_solve_tab(), "4  Solve && Results")

        left = W.QWidget()
        lv = W.QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(self.tabs, 1)
        lv.addWidget(self._build_view_options())
        left.setMinimumWidth(390)
        left.setMaximumWidth(470)

        split = W.QSplitter()
        split.addWidget(left)
        split.addWidget(self.plotter.interactor)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self.progress = W.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Open a G-code file to begin (File > Open G-code…)")
        self._update_enabled()

    # ------------------------------------------------------------------ UI construction

    def _build_actions(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self.act_open = self._act("Open G-code…", self.open_gcode, "Ctrl+O")
        self.act_open_proj = self._act("Open project…", self.open_project)
        self.act_save_proj = self._act("Save project…", self.save_project, "Ctrl+S")
        self.act_save_vtk = self._act("Export results (VTK)…", self.save_vtk)
        self.act_shot = self._act("Save screenshot…", self.save_screenshot)
        for a in (self.act_open, self.act_open_proj, self.act_save_proj):
            fm.addAction(a)
        fm.addSeparator()
        fm.addAction(self.act_save_vtk)
        fm.addAction(self.act_shot)
        fm.addSeparator()
        fm.addAction(self._act("Quit", self.close, "Ctrl+Q"))

        vm = mb.addMenu("&View")
        for label, key in (("Isometric", "iso"), ("Top", "top"), ("Front", "front"), ("Side", "side")):
            vm.addAction(self._act(label, lambda _=False, k=key: self.view.view(k)))
        hm = mb.addMenu("&Help")
        hm.addAction(self._act("About", self.about))

        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.addAction(self.act_open)
        tb.addSeparator()
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for key, label in (("toolpaths", "Toolpaths"), ("model", "Model"), ("results", "Results")):
            a = QtGui.QAction(label, self, checkable=True)
            a.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            self.mode_group.addAction(a)
            tb.addAction(a)
            self.mode_actions[key] = a
        tb.addSeparator()
        for label, key in (("Iso", "iso"), ("Top", "top"), ("Front", "front"), ("Side", "side")):
            tb.addAction(self._act(label, lambda _=False, k=key: self.view.view(k)))

    def _act(self, text, slot, shortcut=None):
        a = QtGui.QAction(text, self)
        a.triggered.connect(lambda _=False: slot())
        if shortcut:
            a.setShortcut(shortcut)
        return a

    def _build_model_tab(self):
        w = W.QWidget()
        lay = W.QVBoxLayout(w)
        self.btn_open = W.QPushButton("Open G-code…")
        self.btn_open.clicked.connect(self.open_gcode)
        lay.addWidget(self.btn_open)
        self.lbl_file = W.QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.lbl_file)

        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("Filament diameter override"))
        self.fil_spin = _dspin(0.0, 5.0, 0.0, 2, 0.05, " mm")
        self.fil_spin.setSpecialValueText("auto")
        self.fil_spin.setToolTip("0 = read from the G-code. Set it if the file has no filament_diameter entry.")
        self.fil_spin.valueChanged.connect(self._filament_changed)
        row.addWidget(self.fil_spin)
        lay.addLayout(row)

        lay.addWidget(W.QLabel("<b>Feature types</b> - checked ones are shown and analysed"))
        self.type_list = W.QListWidget()
        self.type_list.setMaximumHeight(190)
        self.type_list.itemChanged.connect(self._types_changed)
        lay.addWidget(self.type_list)

        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("Colour by"))
        self.color_by = W.QComboBox()
        self.color_by.addItems(["Feature type", "Layer"])
        self.color_by.currentIndexChanged.connect(lambda *_: self.refresh())
        row.addWidget(self.color_by, 1)
        lay.addLayout(row)

        self.layer_label = W.QLabel("Layers shown: -")
        lay.addWidget(self.layer_label)
        self.layer_slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.layer_slider.valueChanged.connect(self._layer_changed)
        lay.addWidget(self.layer_slider)

        box = W.QGroupBox("Analysis model (voxels)")
        form = W.QFormLayout(box)
        self.vox_spin = _dspin(0.1, 5.0, 1.0, 2, 0.1, " mm")
        self.vox_spin.setToolTip("Voxel edge length. ~0.4-0.5 mm resolves single walls; larger = faster, coarser.")
        self.vox_spin.valueChanged.connect(self._update_grid_label)
        form.addRow("Voxel size", self.vox_spin)
        self.minfill_spin = _dspin(0.0, 0.9, 0.05, 2, 0.01)
        self.minfill_spin.setToolTip("Voxels filled less than this fraction are treated as empty.")
        form.addRow("Min. fill to mesh", self.minfill_spin)
        self.grid_label = W.QLabel("")
        self.grid_label.setWordWrap(True)
        form.addRow(self.grid_label)
        self.btn_build = W.QPushButton("Build analysis model")
        self.btn_build.clicked.connect(lambda: self.build_model())
        form.addRow(self.btn_build)
        self.model_label = W.QLabel("")
        self.model_label.setWordWrap(True)
        form.addRow(self.model_label)
        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_material_tab(self):
        self.mat_panel = MaterialPanel()
        self.mat_panel.changed.connect(self._material_changed)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.mat_panel)
        return scroll

    def _build_bc_tab(self):
        w = W.QWidget()
        lay = W.QVBoxLayout(w)
        info = W.QLabel(
            "Click a face in the 3D view to select it (Ctrl+click adds more). Then add it as a support or a load."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        sel = W.QGroupBox("Selection")
        sv = W.QVBoxLayout(sel)
        self.chk_plane = W.QCheckBox("Whole plane (not just the connected flat region)")
        sv.addWidget(self.chk_plane)
        row = W.QHBoxLayout()
        self.face_combo = W.QComboBox()
        self.face_combo.addItems(FACES)
        self.depth_spin = _dspin(0, 1000, 0.0, 2, 0.5, " mm")
        self.depth_spin.setToolTip("Depth of the slab of nodes to select, measured from the model's extreme face")
        b = W.QPushButton("Select model face")
        b.clicked.connect(self.select_face_preset)
        row.addWidget(self.face_combo)
        row.addWidget(self.depth_spin)
        row.addWidget(b)
        sv.addLayout(row)
        grid = W.QGridLayout()
        self.box_spins = {}
        for r, a in enumerate(AXES):
            grid.addWidget(W.QLabel(f"{a.upper()} from"), r, 0)
            lo, hi = _dspin(-1e5, 1e5, 0, 2, 1, " mm"), _dspin(-1e5, 1e5, 0, 2, 1, " mm")
            grid.addWidget(lo, r, 1)
            grid.addWidget(W.QLabel("to"), r, 2)
            grid.addWidget(hi, r, 3)
            self.box_spins[a] = (lo, hi)
        sv.addLayout(grid)
        row = W.QHBoxLayout()
        b1 = W.QPushButton("Select box")
        b1.clicked.connect(self.select_box)
        b2 = W.QPushButton("Clear selection")
        b2.clicked.connect(lambda: self.set_selection(np.empty(0, dtype=int), ""))
        row.addWidget(b1)
        row.addWidget(b2)
        sv.addLayout(row)
        self.sel_label = W.QLabel("Nothing selected")
        self.sel_label.setWordWrap(True)
        sv.addWidget(self.sel_label)
        lay.addWidget(sel)

        sup = W.QGroupBox("Support (fixed displacement)")
        sl = W.QHBoxLayout(sup)
        self.fix_checks = {}
        for a in AXES:
            c = W.QCheckBox(f"fix {a.upper()}")
            c.setChecked(True)
            sl.addWidget(c)
            self.fix_checks[a] = c
        self.btn_support = W.QPushButton("Add support")
        self.btn_support.clicked.connect(self.add_support)
        sl.addWidget(self.btn_support)
        lay.addWidget(sup)

        ld = W.QGroupBox("Load (total force on the selection)")
        lf = W.QFormLayout(ld)
        self.mag_spin = _dspin(0, 1e7, 50.0, 3, 5.0, " N")
        lf.addRow("Magnitude", self.mag_spin)
        self.dir_combo = W.QComboBox()
        self.dir_combo.addItems(["Into the picked face", "Out of the picked face", "+X", "-X", "+Y", "-Y",
                                 "+Z", "-Z", "Custom vector"])
        self.dir_combo.currentIndexChanged.connect(self._dir_changed)
        lf.addRow("Direction", self.dir_combo)
        row = W.QHBoxLayout()
        self.vec_spins = [_dspin(-1e7, 1e7, v, 3, 1.0, " N") for v in (0, 0, -50)]
        for s in self.vec_spins:
            row.addWidget(s)
        self.vec_widget = W.QWidget()
        self.vec_widget.setLayout(row)
        self.vec_widget.hide()
        lf.addRow("Vector", self.vec_widget)
        self.btn_load = W.QPushButton("Add load")
        self.btn_load.clicked.connect(self.add_load)
        lf.addRow(self.btn_load)
        lay.addWidget(ld)

        lay.addWidget(W.QLabel("<b>Defined supports and loads</b>"))
        self.bc_list = W.QListWidget()
        self.bc_list.setMaximumHeight(120)
        self.bc_list.itemSelectionChanged.connect(self._bc_selected)
        lay.addWidget(self.bc_list)
        row = W.QHBoxLayout()
        b3 = W.QPushButton("Remove selected")
        b3.clicked.connect(self.remove_bc)
        b4 = W.QPushButton("Remove all")
        b4.clicked.connect(self.clear_bcs)
        row.addWidget(b3)
        row.addWidget(b4)
        lay.addLayout(row)
        lay.addStretch(1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _build_solve_tab(self):
        w = W.QWidget()
        lay = W.QVBoxLayout(w)
        box = W.QGroupBox("Solve")
        f = W.QFormLayout(box)
        self.solver_combo = W.QComboBox()
        self.solver_combo.addItems(["auto", "direct", "cg"])
        f.addRow("Solver", self.solver_combo)
        self.btn_solve = W.QPushButton("Solve")
        self.btn_solve.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_solve.clicked.connect(self.solve)
        f.addRow(self.btn_solve)
        lay.addWidget(box)

        self.summary = W.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(150)
        self.summary.setPlaceholderText("Solver summary and warnings appear here")
        lay.addWidget(self.summary)

        rb = W.QGroupBox("Results")
        rf = W.QFormLayout(rb)
        self.field_combo = W.QComboBox()
        self.field_combo.addItems([r[0] for r in RESULT_FIELDS])
        self.field_combo.currentIndexChanged.connect(lambda *_: self.refresh())
        rf.addRow("Show", self.field_combo)
        self.clip_check = W.QCheckBox("Clip colour range at 99th percentile")
        self.clip_check.setChecked(True)
        self.clip_check.setToolTip("Stress peaks at supports and load points can hide the rest of the field")
        self.clip_check.toggled.connect(lambda *_: self.refresh())
        rf.addRow(self.clip_check)
        row = W.QHBoxLayout()
        self.scale_spin = _dspin(0, 100000, 1.0, 1, 1.0, " x")
        self.scale_spin.valueChanged.connect(lambda *_: self.refresh())
        b = W.QPushButton("True scale")
        b.clicked.connect(lambda: self.scale_spin.setValue(1.0))
        b2 = W.QPushButton("Auto")
        b2.clicked.connect(self._auto_scale)
        row.addWidget(self.scale_spin)
        row.addWidget(b)
        row.addWidget(b2)
        rf.addRow("Deformation", row)
        self.stale_label = W.QLabel("")
        self.stale_label.setStyleSheet("color: #b45309;")
        self.stale_label.setWordWrap(True)
        rf.addRow(self.stale_label)
        lay.addWidget(rb)
        lay.addStretch(1)
        return w

    def _build_view_options(self):
        box = W.QGroupBox("View")
        lay = W.QVBoxLayout(box)
        row = W.QHBoxLayout()
        self.chk_axes = W.QCheckBox("Axes (mm)")
        self.chk_axes.setChecked(True)
        self.chk_axes.toggled.connect(lambda on: (self.view.set_bounds_visible(on), self.refresh()))
        self.chk_edges = W.QCheckBox("Voxel edges")
        self.chk_edges.toggled.connect(lambda *_: self.refresh())
        self.model_color = W.QComboBox()
        self.model_color.addItems(["Plain", "Fill fraction"])
        self.model_color.currentIndexChanged.connect(lambda *_: self.refresh())
        row.addWidget(self.chk_axes)
        row.addWidget(self.chk_edges)
        row.addWidget(self.model_color)
        lay.addLayout(row)
        row = W.QHBoxLayout()
        self.chk_section = W.QCheckBox("Section")
        self.chk_section.toggled.connect(self._section_changed)
        self.sec_axis = W.QComboBox()
        self.sec_axis.addItems(list("XYZ"))
        self.sec_axis.setCurrentIndex(1)
        self.sec_axis.currentIndexChanged.connect(self._section_axis_changed)
        self.sec_flip = W.QCheckBox("flip")
        self.sec_flip.toggled.connect(self._section_changed)
        self.sec_slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sec_slider.setRange(0, 1000)
        self.sec_slider.setValue(500)
        self.sec_slider.valueChanged.connect(self._section_changed)
        row.addWidget(self.chk_section)
        row.addWidget(self.sec_axis)
        row.addWidget(self.sec_flip)
        row.addWidget(self.sec_slider, 1)
        lay.addLayout(row)
        self.sec_label = W.QLabel("")
        lay.addWidget(self.sec_label)
        self._sec_timer = QtCore.QTimer(self)
        self._sec_timer.setSingleShot(True)
        self._sec_timer.setInterval(60)
        self._sec_timer.timeout.connect(self.refresh)
        return box

    # ------------------------------------------------------------------ state helpers

    def _update_enabled(self):
        have_paths = self.paths is not None
        have_model = self.model is not None
        have_result = self.result is not None
        self.btn_build.setEnabled(have_paths)
        self.btn_solve.setEnabled(have_model)
        self.btn_support.setEnabled(have_model)
        self.btn_load.setEnabled(have_model)
        self.act_save_proj.setEnabled(have_paths)
        self.act_save_vtk.setEnabled(have_result)
        self.mode_actions["toolpaths"].setEnabled(have_paths)
        self.mode_actions["model"].setEnabled(have_model)
        self.mode_actions["results"].setEnabled(have_result)
        self.chk_section.setEnabled(have_model)
        self.mode_actions[self.mode].setChecked(True)

    def set_busy(self, text: str | None):
        busy = text is not None
        self.tabs.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.statusBar().showMessage(text or "Ready")

    def _failed(self, text: str):
        self.set_busy(None)
        W.QMessageBox.critical(self, "threedem", text)

    def set_mode(self, mode: str):
        self.mode = mode
        self._update_enabled()
        self.refresh(reset=True)

    # ------------------------------------------------------------------ G-code and model

    def open_gcode(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Open G-code", str(self.gcode_path.parent) if self.gcode_path
                                                else "", "G-code (*.gcode *.gco *.nc *.txt);;All files (*)")
        if path:
            self.load_gcode(Path(path))

    def load_gcode(self, path: Path, then=None):
        override = self.fil_spin.value() or None
        self.set_busy(f"Reading {path.name}…")

        def done(paths: ToolPaths, _warns):
            self.set_busy(None)
            self._set_paths(path, paths)
            if then:
                then()

        run_job(lambda: parse_gcode(path, override), done, self._failed)

    def _set_paths(self, path: Path, paths: ToolPaths):
        self.gcode_path, self.paths = path, paths
        self.model = self.vox = self.base_quads = self.disp_quads = None
        self.result = self.disp_nodes = None
        self.bcs.clear()
        self.bc_list.clear()
        self.sel_nodes = np.empty(0, dtype=int)
        self.model_label.setText("")
        self.summary.clear()
        if len(paths) == 0:
            W.QMessageBox.warning(self, "threedem", "No extruded moves found in this file.")
        lo, hi = paths.bounds() if len(paths) else (np.zeros(3), np.zeros(3))
        m = paths.meta
        self.lbl_file.setText(
            f"<b>{path.name}</b><br>{len(paths):,} segments, {int(paths.layer.max()) + 1 if len(paths) else 0} layers"
            f"<br>filament {m['filament_diameter']} mm ({m['filament_diameter_source']}), "
            f"{paths.volume.sum():.0f} mm³ extruded"
        )
        self.type_list.blockSignals(True)
        self.type_list.clear()
        pats = [p.lower() for p in DEFAULT_EXCLUDE]
        for i, t in enumerate(paths.types):
            vol = paths.volume[paths.type_id == i].sum()
            item = W.QListWidgetItem(f"{t}   -   {vol:.0f} mm³")
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(*type_color(i)))
            item.setIcon(QtGui.QIcon(pm))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            on = not any(p in t.lower() for p in pats)
            item.setCheckState(QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self.type_list.addItem(item)
        self.type_list.blockSignals(False)
        nl = int(paths.layer.max()) + 1 if len(paths) else 1
        self.layer_slider.blockSignals(True)
        self.layer_slider.setRange(0, max(0, nl - 1))
        self.layer_slider.setValue(max(0, nl - 1))
        self.layer_slider.blockSignals(False)
        self._layer_changed()
        self._update_grid_label()
        self.mode = "toolpaths"
        self._update_enabled()
        self.tabs.setCurrentIndex(0)
        self.setWindowTitle(f"threedem - {path.name}")
        self.refresh(reset=True)
        self._update_box_defaults()

    def checked_types(self) -> list[int]:
        out = []
        for r in range(self.type_list.count()):
            it = self.type_list.item(r)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(it.data(QtCore.Qt.ItemDataRole.UserRole))
        return out

    def _filament_changed(self, *_):
        if self.gcode_path is not None:
            self.load_gcode(self.gcode_path)

    def _types_changed(self, *_):
        self._update_grid_label()
        if self.model is not None:
            self.model_label.setText("<i>Feature selection changed - rebuild the analysis model.</i>")
        if self.mode == "toolpaths":
            self.refresh()

    def _layer_changed(self, *_):
        if self.paths is not None and len(self.paths):
            nl = int(self.paths.layer.max()) + 1
            self.layer_label.setText(f"Layers shown: 1 - {self.layer_slider.value() + 1} of {nl}")
        if self.mode == "toolpaths":
            self.refresh()

    def _update_grid_label(self, *_):
        if self.paths is None or len(self.paths) == 0:
            self.grid_label.setText("")
            return
        part = self.paths.select_types(self.checked_types())
        if len(part) == 0:
            self.grid_label.setText("No feature types selected")
            return
        lo, hi = part.bounds()
        size = self.vox_spin.value()
        dims = np.ceil((hi - lo + float(part.width.max())) / size).astype(int)
        total = int(np.prod(dims))
        txt = f"Grid ≈ {dims[0]} x {dims[1]} x {dims[2]} = {total:,} voxels"
        if total > 3_000_000:
            txt += "  - large: building and solving will be slow; try a bigger voxel size"
        elif total > 600_000:
            txt += "  - fine: solving may take a few minutes"
        self.grid_label.setText(txt)

    def build_model(self, then=None):
        if self.paths is None:
            return
        types = self.checked_types()
        if not types:
            W.QMessageBox.information(self, "threedem", "Select at least one feature type.")
            return
        if self.bcs:
            ans = W.QMessageBox.question(self, "threedem", "Rebuilding clears the supports and loads. Continue?")
            if ans != W.QMessageBox.StandardButton.Yes:
                return
        part, size, minfill = self.paths.select_types(types), self.vox_spin.value(), self.minfill_spin.value()
        material = self.mat_panel.material()  # read widgets here, not in the worker thread
        self.set_busy("Voxelizing toolpaths…")

        def work():
            vox = voxelize(part, size)
            model = Model(vox, material, min_fill=minfill)
            return vox, model, boundary_quads(model)

        def done(res, _w):
            self.set_busy(None)
            self.vox, self.model, self.base_quads = res
            self.bcs.clear()
            self.bc_list.clear()
            self._bc_count = {"support": 0, "load": 0}
            self.sel_nodes = np.empty(0, dtype=int)
            self.result = self.disp_nodes = None
            self.result_stale = False
            self.stale_label.setText("")
            self._section_cache = None
            ne = len(self.model.elements_ijk)
            self.model_label.setText(
                f"{ne:,} elements, {self.model.n_nodes:,} nodes ({3 * self.model.n_nodes:,} DOF); "
                f"{self.vox.fill.sum() * size**3:.0f} mm³ plastic in the grid"
            )
            self._update_box_defaults()
            self.mode = "model"
            self._update_enabled()
            self.tabs.setCurrentIndex(2)
            self.refresh(reset=True)
            self.statusBar().showMessage("Model ready - click a face to select it, then add a support or load")
            if then:
                then()

        run_job(work, done, self._failed)

    def _update_box_defaults(self):
        if self.model is None:
            return
        pts = self.model.node_xyz
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        for a, ax in enumerate(AXES):
            self.box_spins[ax][0].setValue(float(lo[a]))
            self.box_spins[ax][1].setValue(float(hi[a]))

    def _material_changed(self):
        if self.result is not None:
            self.result_stale = True
            self.stale_label.setText("Material changed since the last solve - solve again.")

    # ------------------------------------------------------------------ rendering

    def refresh(self, reset: bool = False):
        if self.mode == "toolpaths" and self.paths is not None:
            by = "layer" if self.color_by.currentIndex() == 1 else "type"
            n = self.view.show_toolpaths(self.paths, self.checked_types(), self.layer_slider.value(), by, reset)
            self.statusBar().showMessage(f"{n:,} toolpath segments shown")
        elif self.model is not None:
            self._render_model(reset)

    def _section_params(self):
        ax = self.sec_axis.currentIndex()
        lo = self.vox.origin[ax]
        hi = lo + self.vox.shape[ax] * self.vox.size
        pos = lo + (hi - lo) * self.sec_slider.value() / 1000.0
        return ax, pos, not self.sec_flip.isChecked()

    def _display_quads(self) -> QuadSet:
        if not self.chk_section.isChecked():
            self.sec_label.setText("")
            return self.base_quads
        ax, pos, below = self._section_params()
        self.sec_label.setText(f"Cut at {AXES[ax].upper()} = {pos:.2f} mm, keeping the "
                               f"{'lower' if below else 'upper'} side")
        key = (ax, round(pos, 4), below)
        if self._section_cache is None or self._section_cache[0] != key:
            q = boundary_quads(self.model, section_mask(self.model, ax, pos, below))
            self._section_cache = (key, q)
        return self._section_cache[1]

    def _section_changed(self, *_):
        if self.model is not None and self.mode != "toolpaths":
            self._sec_timer.start()

    def _section_axis_changed(self, *_):
        self._section_cache = None
        self._section_changed()

    def _current_field(self):
        return RESULT_FIELDS[self.field_combo.currentIndex()]

    def _render_model(self, reset: bool):
        q = self._display_quads()
        self.disp_quads = q
        points = cell = point = None
        title, cmap, clim, ghost = "", "viridis", None, False
        if self.mode == "results" and self.result is not None:
            label, kind, key, cmap = self._current_field()
            scale = self.scale_spin.value()
            points = self.model.node_xyz + scale * self.disp_nodes
            ghost = scale > 0
            title = label
            if kind == "point":
                point = (np.linalg.norm(self.disp_nodes, axis=1) if key == "mag" else self.disp_nodes[:, key])
                vals = point[np.unique(q.nodes)]
            else:
                cell = self.result.cell[key][tuple(q.elem.T)]
                vals = cell
            finite = vals[np.isfinite(vals)]
            if len(finite):
                lo, hi = float(finite.min()), float(finite.max())
                if key == "safety_factor":
                    lo, hi = 0.0, 5.0
                elif self.clip_check.isChecked() and kind == "cell" and key != "safety_factor":
                    hi = float(np.percentile(finite, 99))
                    lo = float(np.percentile(finite, 1)) if lo < 0 else lo
                if hi <= lo:
                    hi = lo + 1e-9
                clim = (lo, hi)
        elif self.model_color.currentIndex() == 1:
            cell = self.vox.fill[tuple(q.elem.T)]
            title, clim = "Fill fraction", (0.0, 1.0)
        self._points = points
        self.view.show_model(self.model, q, cell_values=cell, point_values=point, title=title, cmap=cmap, clim=clim,
                             points=points, edges=self.chk_edges.isChecked(), ghost=ghost, reset=reset)
        self._render_overlays()

    def _render_overlays(self):
        if self.model is None or self.disp_quads is None or self.mode == "toolpaths":
            return
        q, pts = self.disp_quads, self._points
        sel = quads_within(q, self.sel_nodes) if len(self.sel_nodes) else np.empty(0, dtype=int)
        sup = [quads_within(q, b.nodes) for b in self.bcs if b.kind == "support"]
        loads = [(quads_within(q, b.nodes), b.force) for b in self.bcs if b.kind == "load"]
        self.view.show_selection(self.model, q, sel, pts)
        self.view.show_bcs(self.model, q, sup, loads, pts)

    # ------------------------------------------------------------------ selection

    def on_pick(self, quad_id: int, ctrl: bool):
        if self.model is None or self.mode == "toolpaths" or self.disp_quads is None:
            return
        q = self.disp_quads
        region = pick_region(q, quad_id, self.chk_plane.isChecked())
        nodes = nodes_of(q, region)
        axis, sign = int(q.axis[quad_id]), int(q.sign[quad_id])
        self.sel_normal = q.normals()[quad_id].copy()
        coord = float(self.model.node_xyz[q.nodes[quad_id][0], axis])
        area = len(region) * self.vox.size**2
        text = (f"Face {'+' if sign > 0 else '-'}{AXES[axis].upper()} at {AXES[axis].upper()} = {coord:.2f} mm: "
                f"{len(region)} voxel faces (~{area:.0f} mm²), {len(nodes)} nodes")
        if ctrl and len(self.sel_nodes):
            nodes = np.union1d(self.sel_nodes, nodes)
            text += f"  [added; {len(nodes)} nodes selected in total]"
        self.set_selection(nodes, text)
        self.tabs.setCurrentIndex(2)

    def set_selection(self, nodes: np.ndarray, text: str):
        self.sel_nodes = np.asarray(nodes, dtype=int)
        self.sel_label.setText(text or "Nothing selected")
        self._render_overlays()

    def select_face_preset(self):
        if self.model is None:
            return
        try:
            nodes = self.model.select_nodes(face=self.face_combo.currentText(), depth=self.depth_spin.value())
        except ValueError as exc:
            W.QMessageBox.information(self, "threedem", str(exc))
            return
        face = self.face_combo.currentText()
        self.sel_normal = np.zeros(3)
        self.sel_normal[AXES.index(face[0])] = 1.0 if face.endswith("max") else -1.0
        self.set_selection(nodes, f"Model face {face}: {len(nodes)} nodes")

    def select_box(self):
        if self.model is None:
            return
        box = {a: [self.box_spins[a][0].value(), self.box_spins[a][1].value()] for a in AXES}
        try:
            nodes = self.model.select_nodes(box=box)
        except ValueError as exc:
            W.QMessageBox.information(self, "threedem", str(exc))
            return
        self.sel_normal = np.zeros(3)
        self.set_selection(nodes, f"Box selection: {len(nodes)} nodes")

    # ------------------------------------------------------------------ supports and loads

    def _need_selection(self) -> bool:
        if len(self.sel_nodes) == 0:
            W.QMessageBox.information(self, "threedem", "Select a face first (click it in the 3D view).")
            return False
        return True

    def _next_name(self, kind: str) -> str:
        self._bc_count[kind] += 1
        return f"{kind.capitalize()} {self._bc_count[kind]}"

    def add_support(self):
        if not self._need_selection():
            return
        axes = "".join(a for a in AXES if self.fix_checks[a].isChecked())
        if not axes:
            W.QMessageBox.information(self, "threedem", "Tick at least one axis to fix.")
            return
        self.bcs.append(BC("support", self._next_name("support"), self.sel_nodes.copy(), axes=axes))
        self._bcs_changed()

    def _dir_changed(self, *_):
        self.vec_widget.setVisible(self.dir_combo.currentText() == "Custom vector")
        self.mag_spin.setEnabled(self.dir_combo.currentText() != "Custom vector")

    def _load_vector(self) -> np.ndarray | None:
        d = self.dir_combo.currentText()
        if d == "Custom vector":
            return np.array([s.value() for s in self.vec_spins])
        mag = self.mag_spin.value()
        if d in ("Into the picked face", "Out of the picked face"):
            if not np.any(self.sel_normal):
                W.QMessageBox.information(self, "threedem", "Pick a face by clicking it, or choose another direction.")
                return None
            return mag * self.sel_normal * (1 if d.startswith("Out") else -1)
        v = np.zeros(3)
        v[AXES.index(d[1].lower())] = mag * (1 if d[0] == "+" else -1)
        return v

    def add_load(self):
        if not self._need_selection():
            return
        f = self._load_vector()
        if f is None:
            return
        if not np.any(f):
            W.QMessageBox.information(self, "threedem", "The force is zero.")
            return
        self.bcs.append(BC("load", self._next_name("load"), self.sel_nodes.copy(), force=f))
        self._bcs_changed()

    def _bcs_changed(self):
        self.bc_list.blockSignals(True)
        self.bc_list.clear()
        for b in self.bcs:
            self.bc_list.addItem(b.describe())
        self.bc_list.blockSignals(False)
        self.sel_nodes = np.empty(0, dtype=int)
        self.sel_label.setText("Nothing selected")
        if self.result is not None:
            self.result_stale = True
            self.stale_label.setText("Supports/loads changed since the last solve - solve again.")
        self._render_overlays()

    def _bc_selected(self):
        rows = [i.row() for i in self.bc_list.selectedIndexes()]
        if rows:
            b = self.bcs[rows[0]]
            self.sel_nodes = b.nodes
            self.sel_label.setText(f"{b.describe()}")
            self._render_overlays()

    def remove_bc(self):
        rows = sorted({i.row() for i in self.bc_list.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.bcs[r]
        self._bcs_changed()

    def clear_bcs(self):
        self.bcs.clear()
        self._bcs_changed()

    # ------------------------------------------------------------------ solve

    def solve(self):
        if self.model is None:
            return
        if not any(b.kind == "support" for b in self.bcs):
            W.QMessageBox.information(self, "threedem", "Add at least one support (fixed face) first.")
            return
        if not any(b.kind == "load" for b in self.bcs):
            W.QMessageBox.information(self, "threedem", "Add at least one load first.")
            return
        material = self.mat_panel.material()
        try:
            material.local_stiffness()
        except ValueError as exc:
            W.QMessageBox.warning(self, "threedem", f"Invalid material: {exc}")
            return
        model, bcs, p = self.model, list(self.bcs), self.mat_panel.fill_exp.value()
        solver = self.solver_combo.currentText()
        self.summary.setPlainText("Solving…")
        self.set_busy("Solving… (large models can take a few minutes)")

        def work():
            model.clear_bcs()
            model.material = material
            model.fill_exponent = p
            for b in bcs:
                if b.kind == "support":
                    model.fix(b.nodes, b.axes)
                else:
                    model.add_force(b.nodes, b.force)
            return model.solve(solver=solver)

        run_job(work, self._solved, self._solve_failed)

    def _solve_failed(self, text: str):
        self.summary.setPlainText(f"Solve failed:\n{text}")
        self._failed(text)

    def _solved(self, result: Result, warns: list[str]):
        self.set_busy(None)
        self.result = result
        self.disp_nodes = node_displacement(self.model, result.displacement)
        self.result_stale = False
        self.stale_label.setText("")
        text = result.summary()
        if warns:
            text += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in dict.fromkeys(warns))
        self.summary.setPlainText(text)
        self._auto_scale(refresh=False)
        self.mode = "results"
        self._update_enabled()
        self.tabs.setCurrentIndex(3)
        self.refresh(reset=False)
        self.statusBar().showMessage("Solved")

    def _auto_scale(self, refresh: bool = True):
        if self.disp_nodes is None:
            return
        dmax = float(np.linalg.norm(self.disp_nodes, axis=1).max())
        extent = float(np.max(self.model.node_xyz.max(axis=0) - self.model.node_xyz.min(axis=0)))
        scale = max(1.0, 0.08 * extent / dmax) if dmax > 0 else 1.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(round(scale, 1))
        self.scale_spin.blockSignals(False)
        if refresh:
            self.refresh()

    # ------------------------------------------------------------------ files

    def save_vtk(self):
        if self.result is None:
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Export results", "result.vtk", "VTK (*.vtk)")
        if path:
            write_vtk(path, self.vox, self.result)
            self.statusBar().showMessage(f"Wrote {path}")

    def save_screenshot(self):
        path, _ = W.QFileDialog.getSaveFileName(self, "Save screenshot", "threedem.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)

    def save_project(self):
        if self.paths is None:
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Save project", str(self.gcode_path.with_suffix(".threedem")),
                                                "threedem project (*.threedem)")
        if not path:
            return
        names = [self.paths.types[i] for i in self.checked_types()]
        bcs = []
        for b in self.bcs:
            d = {"kind": b.kind, "name": b.name, "points": self.model.node_xyz[b.nodes].round(5).tolist()}
            if b.kind == "support":
                d["axes"] = b.axes
            else:
                d["force"] = [float(x) for x in b.force]
            bcs.append(d)
        data = {
            "version": 1,
            "gcode": str(self.gcode_path),
            "filament_diameter_override": self.fil_spin.value(),
            "types": names,
            "voxel_size": self.vox_spin.value(),
            "min_fill": self.minfill_spin.value(),
            "fill_exponent": self.mat_panel.fill_exp.value(),
            "material": asdict(self.mat_panel.material()),
            "bcs": bcs if self.model is not None else [],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        self.statusBar().showMessage(f"Saved {path}")

    def open_project(self, path: str | None = None):
        if not path:
            path, _ = W.QFileDialog.getOpenFileName(self, "Open project", "", "threedem project (*.threedem)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            gcode = Path(data["gcode"])
            if not gcode.exists():  # moved next to the project file?
                alt = Path(path).parent / gcode.name
                gcode = alt if alt.exists() else gcode
            if not gcode.exists():
                raise FileNotFoundError(f"G-code not found: {gcode}")
        except Exception as exc:
            W.QMessageBox.warning(self, "Open project", f"Could not open the project:\n{exc}")
            return
        self.fil_spin.blockSignals(True)
        self.fil_spin.setValue(data.get("filament_diameter_override", 0.0))
        self.fil_spin.blockSignals(False)
        self.vox_spin.setValue(data.get("voxel_size", 1.0))
        self.minfill_spin.setValue(data.get("min_fill", 0.05))
        self.mat_panel.set_material(get_material(data["material"]), None)
        self.mat_panel.fill_exp.setValue(data.get("fill_exponent", 1.0))

        def after_model():
            tree = cKDTree(self.model.node_xyz)
            for d in data.get("bcs", []):
                _, nodes = tree.query(np.array(d["points"]))
                nodes = np.unique(nodes)
                if d["kind"] == "support":
                    self.bcs.append(BC("support", d["name"], nodes, axes=d["axes"]))
                else:
                    self.bcs.append(BC("load", d["name"], nodes, force=np.array(d["force"])))
                self._bc_count[d["kind"]] += 1
            self._bcs_changed()

        def after_paths():
            wanted = set(data.get("types", []))
            self.type_list.blockSignals(True)
            for r in range(self.type_list.count()):
                it = self.type_list.item(r)
                name = self.paths.types[it.data(QtCore.Qt.ItemDataRole.UserRole)]
                it.setCheckState(QtCore.Qt.CheckState.Checked if name in wanted else QtCore.Qt.CheckState.Unchecked)
            self.type_list.blockSignals(False)
            self._update_grid_label()
            self.build_model(then=after_model if data.get("bcs") else None)

        self.load_gcode(gcode, then=after_paths)

    def about(self):
        W.QMessageBox.about(
            self, "threedem",
            "<b>threedem</b> - finite element analysis of 3D-printed parts from slicer G-code.<br><br>"
            "Early alpha. Material values are illustrative; results are not validated against physical tests.",
        )

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    app = W.QApplication.instance() or W.QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if argv:
        p = Path(argv[0])
        if p.suffix.lower() == ".threedem":
            win.open_project(str(p))
        elif p.exists():
            win.load_gcode(p)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
