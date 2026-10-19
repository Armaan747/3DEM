"""Drive the real desktop window end to end.

Needs a display and the GUI extras, so it is opt-in:  THREEDEM_GUI_TESTS=1 pytest tests/test_gui_smoke.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("THREEDEM_GUI_TESTS") != "1",
                                reason="set THREEDEM_GUI_TESTS=1 to run GUI tests (needs a display)")

GEOM = Path(__file__).parent / "Geom"


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyvistaqt")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def wait(app, cond, timeout=120):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, "timed out"


def test_full_workflow(qapp, tmp_path, monkeypatch):
    from PySide6 import QtCore, QtWidgets as W
    from PySide6.QtTest import QTest

    from threedem.gui.app import MainWindow

    popups = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(W.QMessageBox, name, staticmethod(lambda *a, **k: popups.append(a[2])))

    win = MainWindow()
    win.show()
    wait(qapp, win.isVisible)
    win.load_gcode(next(GEOM.glob("*.gcode")))
    wait(qapp, lambda: win.paths is not None)
    assert "Skirt" in win.paths.types
    # skirt / custom start unchecked
    checked = {win.paths.types[i] for i in win.checked_types()}
    assert "Skirt" not in checked and "Outer wall" in checked

    win.vox_spin.setValue(1.0)
    win.build_model()
    wait(qapp, lambda: win.model is not None)

    # support on the far half of the base via the box tool
    win.box_spins["z"][0].setValue(0.0)
    win.box_spins["z"][1].setValue(0.0)
    win.box_spins["y"][0].setValue(200.0)
    win.select_box()
    win.add_support()
    # load on the top of the upright, chosen with a *real* mouse click
    win.view.view("iso")
    q = win.disp_quads
    top = np.flatnonzero((q.axis == 2) & (q.sign > 0))
    win.on_pick(int(top[np.argmax(q.plane[top])]), False)
    win.dir_combo.setCurrentText("+Y")
    win.mag_spin.setValue(50.0)
    win.add_load()
    assert len(win.bcs) == 2

    win.solve()
    wait(qapp, lambda: win.result is not None)
    assert win.result.reaction[1] == pytest.approx(-50.0, abs=1e-3)
    assert win.mode == "results"
    for i in range(win.field_combo.count()):  # every result field renders without error
        win.field_combo.setCurrentIndex(i)
        qapp.processEvents()

    # project round trip restores the boundary conditions
    proj = tmp_path / "p.threedem"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(proj), "")))
    win.save_project()
    win2 = MainWindow()
    win2.show()
    win2.open_project(str(proj))
    wait(qapp, lambda: win2.model is not None and len(win2.bcs) == 2)
    assert {b.kind for b in win2.bcs} == {"support", "load"}
    win2.close()
    win.close()
    assert popups == []


def test_click_selects_a_face_and_drag_does_not(qapp):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    from threedem.gui.app import MainWindow

    win = MainWindow()
    win.show()
    wait(qapp, win.isVisible)
    win.load_gcode(next(GEOM.glob("*.gcode")))
    wait(qapp, lambda: win.paths is not None)
    win.build_model()
    wait(qapp, lambda: win.model is not None)
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.03)
    w = win.plotter.interactor
    c = w.rect().center()
    QTest.mouseClick(w, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, c)
    qapp.processEvents()
    assert len(win.sel_nodes) > 0
    win.set_selection(np.empty(0, dtype=int), "")
    QTest.mousePress(w, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, c)
    QTest.mouseMove(w, c + QtCore.QPoint(60, 20))
    QTest.mouseRelease(w, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                       c + QtCore.QPoint(60, 20))
    qapp.processEvents()
    assert len(win.sel_nodes) == 0
    win.close()
