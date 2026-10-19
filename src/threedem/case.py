"""Run an analysis described by a JSON case file.

Example::

    {
      "gcode": "bracket.gcode",
      "voxel_size": 0.5,
      "material": "PLA",
      "exclude_types": ["skirt", "brim", "support", "custom"],
      "supports": [{"face": "zmin", "axes": "xyz"}],
      "loads": [{"box": {"z": [19.5, null]}, "force": [0, 0, -50]}],
      "output": "bracket_result.vtk"
    }

``box`` is ``{"x": [lo, hi], ...}`` in G-code coordinates (null = open) and
``face`` is one of xmin/xmax/ymin/ymax/zmin/zmax.  Forces are in N, lengths in mm.
Relative paths are resolved next to the case file.
"""

from __future__ import annotations

import json
from pathlib import Path

from .gcode import DEFAULT_EXCLUDE, parse_gcode
from .solve import Model, Result
from .vtkio import write_vtk
from .voxelize import VoxelModel, voxelize


def load_case(path) -> tuple[dict, Path]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path.parent


def run_case(case: dict, base_dir=".", verbose: bool = True) -> tuple[VoxelModel, Model, Result]:
    base = Path(base_dir)
    log = print if verbose else (lambda *a, **k: None)

    paths = parse_gcode(base / case["gcode"], case.get("filament_diameter"))
    paths = paths.exclude(case.get("exclude_types", DEFAULT_EXCLUDE))
    log(paths.summary())

    vox = voxelize(paths, float(case.get("voxel_size", 0.5)))
    log(vox)
    model = Model(
        vox,
        case.get("material", "PLA"),
        min_fill=float(case.get("min_fill", 0.05)),
        fill_exponent=float(case.get("fill_exponent", 1.0)),
    )

    for s in case.get("supports", []):
        nodes = model.select_nodes(s.get("box"), s.get("face"), float(s.get("depth", 0.0)))
        model.fix(nodes, s.get("axes", "xyz"), float(s.get("value", 0.0)))
        log(f"support: {len(nodes)} nodes, axes {s.get('axes', 'xyz')}")
    for ld in case.get("loads", []):
        nodes = model.select_nodes(ld.get("box"), ld.get("face"), float(ld.get("depth", 0.0)))
        model.add_force(nodes, ld["force"])
        log(f"load: {ld['force']} N on {len(nodes)} nodes")

    result = model.solve(solver=case.get("solver", "auto"))
    log(result.summary())
    out = case.get("output")
    if out:
        out_path = base / out
        write_vtk(out_path, vox, result)
        log(f"wrote {out_path}")
    return vox, model, result
