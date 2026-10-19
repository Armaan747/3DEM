"""Regression tests on a real OrcaSlicer 2.3.1 file (L bracket, Klipper flavour, 0.2 mm PLA).

The part is a 30 x 60 x 10 mm base plate with a 10 mm thick, 25 mm tall upright at the
y-min end, printed flat with 3 walls and 10 % rectilinear infill.
"""

import json
from pathlib import Path

import pytest

from threedem import parse_gcode, run_case

GEOM = Path(__file__).parent / "Geom"
GCODE = GEOM / "L Bracket Test 1_0.2mm_PLA_Elegoo Neptune 4 Max_21m4s.gcode"


@pytest.fixture(scope="module")
def paths():
    return parse_gcode(GCODE)


def test_parses_orca_output(paths):
    assert paths.layer.max() + 1 == 126
    assert paths.meta["filament_diameter"] == 1.75
    assert paths.meta["filament_diameter_source"] == "file"
    assert {"Outer wall", "Inner wall", "Sparse infill", "Top surface", "Skirt", "Custom"} <= set(paths.types)


def test_structural_part_after_exclusions(paths):
    part = paths.exclude()
    assert not {"Skirt", "Custom"} & {part.types[i] for i in set(part.type_id)}
    lo, hi = part.bounds()
    size = hi - lo
    assert size[0] == pytest.approx(29.6, abs=0.3)  # centreline extents: 30 mm minus one road width
    assert size[1] == pytest.approx(59.6, abs=0.3)
    assert size[2] == pytest.approx(24.8, abs=0.3)
    assert part.volume.sum() == pytest.approx(8247, rel=0.01)


def test_l_bracket_case_runs_and_balances():
    case = json.loads((GEOM / "l_bracket_case.json").read_text())
    case["output"] = None  # do not leave a VTK file behind
    _, _, res = run_case(case, GEOM, verbose=False)
    assert res.applied[1] == pytest.approx(50.0)
    assert res.reaction[1] == pytest.approx(-50.0, abs=1e-3)
    assert 0 < res.max_displacement() < 5
