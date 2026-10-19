"""Command-line interface: ``threedem info | run | demo``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .case import load_case, run_case
from .gcode import DEFAULT_EXCLUDE, parse_gcode
from .synth import bar_gcode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="threedem", description="Finite element analysis of 3D-printed parts from G-code.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="summarise a G-code file (bounds, feature types, plastic volume)")
    pi.add_argument("gcode")
    pi.add_argument("--filament-diameter", type=float)

    pr = sub.add_parser("run", help="run an analysis from a JSON case file")
    pr.add_argument("case")

    pd = sub.add_parser("demo", help="write and solve a small cantilever example in a folder")
    pd.add_argument("folder", nargs="?", default="threedem_demo")

    pg = sub.add_parser("gui", help="open the desktop app (needs: pip install threedem[gui])")
    pg.add_argument("file", nargs="?", help="optional .gcode or .threedem project to open")

    args = p.parse_args(argv)
    if args.cmd == "gui":
        try:
            from .gui.app import main as gui_main
        except ImportError as exc:
            print(f"The GUI needs extra packages ({exc}).\nInstall them with:  pip install \"threedem[gui]\"",
                  file=sys.stderr)
            return 1
        return gui_main([args.file] if args.file else [])
    if args.cmd == "info":
        tp = parse_gcode(args.gcode, args.filament_diameter)
        print(tp.summary())
        print(f"filament diameter: {tp.meta['filament_diameter']} mm ({tp.meta['filament_diameter_source']})")
        kept = tp.exclude(DEFAULT_EXCLUDE)
        print(f"after default exclusions {DEFAULT_EXCLUDE}: {kept.volume.sum():.1f} mm^3 remain")
    elif args.cmd == "run":
        case, base = load_case(args.case)
        run_case(case, base)
    elif args.cmd == "demo":
        folder = Path(args.folder)
        folder.mkdir(parents=True, exist_ok=True)
        text, dims = bar_gcode(length=40, width=8, height=8, pattern="alt")
        (folder / "bar.gcode").write_text(text)
        case = {
            "gcode": "bar.gcode",
            "voxel_size": 0.5,
            "material": "PLA",
            "supports": [{"face": "xmin"}],
            "loads": [{"face": "xmax", "force": [0, 0, -20]}],
            "output": "bar_result.vtk",
        }
        (folder / "bar_case.json").write_text(json.dumps(case, indent=2))
        print(f"demo: {dims['length']} x {dims['width']} x {dims['height']} mm bar, fixed at x-min, 20 N down at x-max")
        run_case(case, folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
