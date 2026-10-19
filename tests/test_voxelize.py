import numpy as np
import pytest

from threedem.gcode import parse_gcode
from threedem.synth import bar_gcode
from threedem.voxelize import voxelize


def test_plastic_volume_is_conserved():
    text, dims = bar_gcode(length=10, width=4, height=2)
    tp = parse_gcode(text)
    vox = voxelize(tp, 0.5, origin=(0, 0, 0), shape=(20, 8, 4))
    assert vox.fill.sum() * 0.5**3 == pytest.approx(tp.volume.sum(), rel=1e-3)
    assert vox.meta["plastic_in_grid"] == pytest.approx(tp.volume.sum(), rel=1e-6)


def test_solid_bar_fills_aligned_grid():
    text, _ = bar_gcode(length=10, width=4, height=2)
    vox = voxelize(parse_gcode(text), 0.5, origin=(0, 0, 0), shape=(20, 8, 4))
    assert vox.fill.min() > 0.99


@pytest.mark.parametrize("pattern,c2,c4", [("x", 1.0, 1.0), ("y", -1.0, 1.0), ("alt", 0.0, 1.0)])
def test_orientation_harmonics(pattern, c2, c4):
    text, _ = bar_gcode(length=6, width=4, height=2, pattern=pattern)
    vox = voxelize(parse_gcode(text), 1.0, origin=(0, 0, 0), shape=(6, 4, 2))
    mean = vox.orient.reshape(-1, 4).mean(axis=0)
    assert mean[0] == pytest.approx(c2, abs=1e-3)  # cos 2t
    assert mean[1] == pytest.approx(0.0, abs=1e-3)  # sin 2t
    assert mean[2] == pytest.approx(c4, abs=1e-3)  # cos 4t


def test_empty_paths_rejected():
    tp = parse_gcode("G1 X1 Y1\n")
    with pytest.raises(ValueError):
        voxelize(tp, 0.5)
