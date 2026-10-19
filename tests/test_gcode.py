import math

import numpy as np
import pytest

from threedem.gcode import parse_gcode
from threedem.synth import bar_gcode


def test_synthetic_bar_volume_matches_geometry():
    text, dims = bar_gcode(length=10, width=4, height=2, pitch=0.4)
    tp = parse_gcode(text)
    assert tp.volume.sum() == pytest.approx(dims["length"] * dims["width"] * dims["height"], rel=1e-4)
    assert tp.meta["filament_diameter"] == 1.75
    assert tp.meta["filament_diameter_source"] == "file"
    assert tp.layer.max() + 1 == dims["layers"]


def test_width_and_height_comments_are_used():
    text, _ = bar_gcode(length=4, width=2, height=0.4, pitch=0.4, layer_height=0.2)
    tp = parse_gcode(text)
    assert np.allclose(tp.height, 0.2)
    assert np.allclose(tp.width, 0.4 + 0.2 * (1 - math.pi / 4), atol=1e-4)


def test_width_derived_from_extrusion_when_no_comments():
    text, _ = bar_gcode(length=4, width=2, height=0.4, pitch=0.4, layer_height=0.2)
    stripped = "\n".join(l for l in text.splitlines() if not l.startswith((";WIDTH", ";HEIGHT")))
    tp = parse_gcode(stripped)
    assert np.allclose(tp.height, 0.2)  # inferred from Z steps
    assert np.allclose(tp.width, 0.4 + 0.2 * (1 - math.pi / 4), atol=1e-3)


def test_retraction_is_not_deposited():
    g = """
M83
G1 Z0.2
G1 X0 Y0
G1 E0.8 ; unretract
G1 X10 Y0 E0.5
G1 E-0.8 ; retract
G1 X20 Y0 ; travel
G1 E0.8 ; unretract
G1 X30 Y0 E0.5
"""
    tp = parse_gcode(g, filament_diameter=1.75)
    area = math.pi / 4 * 1.75**2
    assert tp.volume.sum() == pytest.approx(1.0 * area)  # only the two 0.5 mm printing moves
    assert len(tp) == 2


def test_absolute_extrusion_and_g92():
    g = """
M82
G92 E0
G1 Z0.2
G1 X10 Y0 E1.0
G92 E0
G1 X20 Y0 E1.0
"""
    tp = parse_gcode(g, filament_diameter=1.75)
    assert tp.volume.sum() == pytest.approx(2.0 * math.pi / 4 * 1.75**2)


def test_arc_length_and_volume():
    # half circle of radius 5 -> length pi*5
    g = "M83\nG1 Z0.2\nG1 X0 Y0\nG3 X0 Y10 I0 J5 E1.0\n"
    tp = parse_gcode(g, filament_diameter=1.75)
    assert tp.length.sum() == pytest.approx(math.pi * 5, rel=0.01)
    assert tp.volume.sum() == pytest.approx(math.pi / 4 * 1.75**2)
    # a counter-clockwise arc from (0,0) to (0,10) with centre (0,5) bulges towards +x
    assert tp.p1[:, 0].max() == pytest.approx(5.0, abs=0.05)


def test_exclude_types():
    g = """
M83
G1 Z0.2
;TYPE:Skirt
G1 X0 Y0
G1 X10 Y0 E1
;TYPE:Outer wall
G1 X10 Y10 E1
"""
    tp = parse_gcode(g, filament_diameter=1.75)
    assert set(tp.types) == {"Skirt", "Outer wall"}
    kept = tp.exclude()
    assert len(kept) == 1 and kept.types[kept.type_id[0]] == "Outer wall"


def test_binary_gcode_rejected(tmp_path):
    f = tmp_path / "x.bgcode"
    f.write_bytes(b"GCDE\x01\x00\x00\x00")
    with pytest.raises(ValueError, match="binary"):
        parse_gcode(f)
