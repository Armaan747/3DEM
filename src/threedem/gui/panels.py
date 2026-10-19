"""Reusable Qt panels."""

from __future__ import annotations

import json
from dataclasses import asdict, fields

from PySide6 import QtCore, QtWidgets as W

from ..materials import PRESETS, Material, get_material

# attribute, label, unit, minimum, maximum, decimals, tooltip
_STIFFNESS = [
    ("E1", "E1  along the road", "MPa", 1, 1e6, 0, "Young's modulus along the printed line (raster direction)"),
    ("E2", "E2  across the road", "MPa", 1, 1e6, 0, "Young's modulus across the line, in the layer plane (road-to-road bond)"),
    ("E3", "E3  build direction Z", "MPa", 1, 1e6, 0, "Young's modulus through the layers (layer-to-layer bond) - usually the lowest"),
    ("G12", "G12  in-plane shear", "MPa", 1, 1e6, 0, "Shear modulus in the layer plane"),
    ("G13", "G13  shear 1-Z", "MPa", 1, 1e6, 0, "Shear modulus between the line direction and Z"),
    ("G23", "G23  shear 2-Z", "MPa", 1, 1e6, 0, "Shear modulus between the cross direction and Z"),
    ("nu12", "nu12", "", 0.0, 0.49, 3, "Poisson ratio 1-2"),
    ("nu13", "nu13", "", 0.0, 0.49, 3, "Poisson ratio 1-Z"),
    ("nu23", "nu23", "", 0.0, 0.49, 3, "Poisson ratio 2-Z"),
]
_STRENGTH = [
    ("strength_1", "Tensile, along road", "MPa", 0.1, 1e4, 1, "Tensile strength along the line"),
    ("strength_2", "Tensile, across road", "MPa", 0.1, 1e4, 1, "Tensile strength across the line, in plane"),
    ("strength_3", "Tensile, across layers", "MPa", 0.1, 1e4, 1, "Interlayer tensile strength (delamination)"),
    ("shear_strength_3", "Shear, across layers", "MPa", 0.1, 1e4, 1, "Interlayer shear strength"),
]


class MaterialPanel(W.QWidget):
    """Preset chooser plus editable orthotropic constants.  Emits ``changed`` on any edit."""

    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._spins: dict[str, W.QDoubleSpinBox] = {}
        self._loading = False
        lay = W.QVBoxLayout(self)

        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("Preset"))
        self.preset = W.QComboBox()
        self.preset.addItems([n for n in PRESETS] + ["Custom"])
        self.preset.currentTextChanged.connect(self._preset_changed)
        row.addWidget(self.preset, 1)
        lay.addLayout(row)

        note = W.QLabel(
            "Printed plastic is orthotropic. Axis 1 runs along each printed line, 2 across it in the layer "
            "plane, 3 is the build direction. Preset numbers are <b>illustrative only</b> - enter values "
            "from tensile tests of your own prints."
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        lay.addWidget(self._group("Stiffness", _STIFFNESS))
        lay.addWidget(self._group("Strength (for the safety factor)", _STRENGTH))

        adv = W.QGroupBox("Homogenization")
        form = W.QFormLayout(adv)
        self.fill_exp = W.QDoubleSpinBox()
        self.fill_exp.setRange(0.5, 4.0)
        self.fill_exp.setSingleStep(0.1)
        self.fill_exp.setValue(1.0)
        self.fill_exp.setToolTip("Stiffness of a partly filled voxel scales with fill^p. 1 = rule of mixtures.")
        self.fill_exp.valueChanged.connect(lambda *_: self.changed.emit())
        form.addRow("Fill exponent p", self.fill_exp)
        lay.addWidget(adv)

        io = W.QHBoxLayout()
        b_save, b_load = W.QPushButton("Save material…"), W.QPushButton("Load material…")
        b_save.clicked.connect(self.save)
        b_load.clicked.connect(self.load)
        io.addWidget(b_save)
        io.addWidget(b_load)
        lay.addLayout(io)
        lay.addStretch(1)
        self.set_material(PRESETS["PLA"], "PLA")

    def _group(self, title, spec):
        box = W.QGroupBox(title)
        form = W.QFormLayout(box)
        for attr, label, unit, lo, hi, dec, tip in spec:
            sp = W.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSuffix(f" {unit}" if unit else "")
            sp.setToolTip(tip)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._edited)
            form.addRow(label, sp)
            self._spins[attr] = sp
        return box

    def _edited(self, *_):
        if self._loading:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText("Custom")
        self.preset.blockSignals(False)
        self.changed.emit()

    def _preset_changed(self, name: str):
        if name in PRESETS:
            self.set_material(PRESETS[name], name)
            self.changed.emit()

    def set_material(self, m: Material, preset: str | None = None):
        self._loading = True
        for attr, sp in self._spins.items():
            sp.setValue(float(getattr(m, attr)))
        self._loading = False
        self.preset.blockSignals(True)
        self.preset.setCurrentText(preset or "Custom")
        self.preset.blockSignals(False)
        self._name = m.name

    def material(self) -> Material:
        vals = {a: sp.value() for a, sp in self._spins.items()}
        return Material(name=self.preset.currentText(), **vals)

    def save(self):
        path, _ = W.QFileDialog.getSaveFileName(self, "Save material", "material.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.material()), f, indent=2)

    def load(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Load material", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            known = {f_.name for f_ in fields(Material)}
            m = get_material({k: v for k, v in spec.items() if k in known})
        except Exception as exc:
            W.QMessageBox.warning(self, "Load material", f"Could not read material:\n{exc}")
            return
        self.set_material(m, None)
        self.changed.emit()
