"""G-code parser: turns a sliced file into a set of extruded toolpath segments.

Only the physically relevant parts of G-code are interpreted: linear and arc
moves, positioning/extrusion modes, ``G92`` E resets and retraction.  Slicer
comments (``;TYPE:``, ``;WIDTH:``, ``;HEIGHT:`` and the ``; key = value``
configuration block) are used when present, so output from OrcaSlicer, Bambu
Studio and PrusaSlicer works best, but plain Marlin-style G-code also loads.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Feature types that normally do not carry structural load.  Matched as
# case-insensitive substrings of the ``;TYPE:`` comment.
DEFAULT_EXCLUDE = ("skirt", "brim", "support", "custom", "prime", "wipe")

_CONFIG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]*?)\s*=\s*(.+?)\s*$")
_WANTED_CONFIG = {"filament_diameter", "layer_height", "nozzle_diameter"}
_ARC_CHORD = 0.4  # mm, maximum chord length when discretising G2/G3 arcs


@dataclass
class ToolPaths:
    """Extruded segments.  Arrays all have length ``n``; lengths in mm, volume in mm^3."""

    p0: np.ndarray  # (n, 3) start points
    p1: np.ndarray  # (n, 3) end points
    volume: np.ndarray  # deposited plastic volume of each segment
    width: np.ndarray  # road width
    height: np.ndarray  # layer height
    type_id: np.ndarray  # index into ``types``
    layer: np.ndarray  # layer index (0-based)
    types: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.volume)

    @property
    def length(self) -> np.ndarray:
        return np.linalg.norm(self.p1 - self.p0, axis=1)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the segment end points (not padded by road width)."""
        pts = np.concatenate([self.p0, self.p1])
        return pts.min(axis=0), pts.max(axis=0)

    def exclude(self, patterns=DEFAULT_EXCLUDE) -> "ToolPaths":
        """Drop segments whose feature type contains any of ``patterns`` (case-insensitive)."""
        pats = [p.lower() for p in patterns]
        drop = np.array([any(p in t.lower() for p in pats) for t in self.types], dtype=bool)
        if not drop.any():
            return self
        keep = ~drop[self.type_id]
        return self._mask(keep)

    def _mask(self, keep: np.ndarray) -> "ToolPaths":
        return ToolPaths(
            self.p0[keep], self.p1[keep], self.volume[keep], self.width[keep],
            self.height[keep], self.type_id[keep], self.layer[keep], self.types, self.meta,
        )

    def summary(self) -> str:
        if len(self) == 0:
            return "no extruded segments"
        lo, hi = self.bounds()
        lines = [
            f"segments:  {len(self)}",
            f"layers:    {int(self.layer.max()) + 1}",
            f"bounds:    x {lo[0]:.2f}..{hi[0]:.2f}  y {lo[1]:.2f}..{hi[1]:.2f}  z {lo[2]:.2f}..{hi[2]:.2f} mm",
            f"plastic:   {self.volume.sum():.1f} mm^3",
            f"road:      width ~{np.median(self.width):.2f} mm, height ~{np.median(self.height):.2f} mm",
            "volume by feature type:",
        ]
        for i, t in enumerate(self.types):
            v = self.volume[self.type_id == i].sum()
            if v > 0:
                lines.append(f"  {t:<28s}{v:10.1f} mm^3")
        return "\n".join(lines)


def parse_gcode(source, filament_diameter: float | None = None) -> ToolPaths:
    """Parse a G-code file path or an iterable of lines.

    ``filament_diameter`` overrides the value found in the file's config block
    (default 1.75 mm if neither is available).
    """
    if isinstance(source, (str, Path)) and "\n" not in str(source):
        path = Path(source)
        with open(path, "rb") as fb:
            if fb.read(4) == b"GCDE":
                raise ValueError(
                    "binary G-code (.bgcode) is not supported; export ASCII G-code from the slicer"
                )
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return _parse_lines(f, filament_diameter)
    if isinstance(source, str):
        source = source.splitlines()
    return _parse_lines(source, filament_diameter)


def _parse_lines(lines, filament_diameter) -> ToolPaths:
    x = y = z = 0.0
    e_prev = 0.0
    rel_xyz = False
    rel_e = False
    retracted = 0.0  # filament pulled back and not yet recovered
    cur_type = "unknown"
    cur_width = math.nan
    cur_height_c = math.nan
    layer_z = 0.0
    layer_h = math.nan
    layer = -1
    config: dict[str, str] = {}

    types: list[str] = []
    type_index: dict[str, int] = {}
    p0, p1, de, wid, hgt, tid, lay = [], [], [], [], [], [], []

    def type_id(name: str) -> int:
        i = type_index.get(name)
        if i is None:
            i = type_index[name] = len(types)
            types.append(name)
        return i

    for raw in lines:
        code, sep, comment = raw.partition(";")
        if sep:
            c = comment.strip()
            if c.startswith("TYPE:"):
                cur_type = c[5:].strip() or "unknown"
            elif c.startswith("WIDTH:"):
                cur_width = _num(c[6:], math.nan)
            elif c.startswith("HEIGHT:"):
                cur_height_c = _num(c[7:], math.nan)
            else:
                m = _CONFIG_RE.match(c)
                if m and m.group(1).strip() in _WANTED_CONFIG:
                    config[m.group(1).strip()] = m.group(2)
        tokens = code.split()
        if not tokens:
            continue
        cmd = tokens[0].upper()

        if cmd in ("G0", "G1", "G00", "G01", "G2", "G3", "G02", "G03"):
            args = _args(tokens[1:])
            nx = _target(x, args.get("X"), rel_xyz)
            ny = _target(y, args.get("Y"), rel_xyz)
            nz = _target(z, args.get("Z"), rel_xyz)
            if "E" in args:
                e_val = args["E"]
                d_e = e_val if rel_e else e_val - e_prev
                e_prev = e_prev + e_val if rel_e else e_val
            else:
                d_e = 0.0

            extruded = 0.0
            if d_e < 0:
                retracted += -d_e
            elif d_e > 0:
                recovered = min(retracted, d_e)
                retracted -= recovered
                extruded = d_e - recovered

            moved_xy = (nx - x) ** 2 + (ny - y) ** 2 > 1e-12
            if extruded > 0 and moved_xy:
                if cmd in ("G2", "G3", "G02", "G03"):
                    pts = _arc_points(x, y, z, nx, ny, nz, args, cw=cmd.endswith("2"))
                else:
                    pts = [(x, y, z), (nx, ny, nz)]
                seg_len = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                total = sum(seg_len)
                if total > 0:
                    if abs(nz - layer_z) > 1e-6:
                        dz = nz - layer_z
                        layer_h = dz if 0.02 < dz < 2.0 else math.nan
                        layer_z = nz
                        layer += 1
                    elif layer < 0:
                        layer = 0
                    h = cur_height_c if not math.isnan(cur_height_c) else layer_h
                    ti = type_id(cur_type)
                    for i, sl in enumerate(seg_len):
                        if sl <= 0:
                            continue
                        p0.append(pts[i])
                        p1.append(pts[i + 1])
                        de.append(extruded * sl / total)
                        wid.append(cur_width)
                        hgt.append(h)
                        tid.append(ti)
                        lay.append(layer)
            x, y, z = nx, ny, nz
        elif cmd == "G90":
            rel_xyz = False
            # Many firmwares tie E mode to G90/G91 only when M82/M83 was never used;
            # slicers always emit M82/M83 explicitly, so E is left alone.
        elif cmd == "G91":
            rel_xyz = True
        elif cmd == "M82":
            rel_e = False
        elif cmd == "M83":
            rel_e = True
        elif cmd == "G92":
            args = _args(tokens[1:])
            if "E" in args:
                e_prev = args["E"]
            x = args.get("X", x)
            y = args.get("Y", y)
            z = args.get("Z", z)

    fil_d = filament_diameter
    source = "argument"
    if fil_d is None:
        fil_d = _first_number(config.get("filament_diameter"))
        source = "file"
    if fil_d is None:
        fil_d, source = 1.75, "default"
    cfg_h = _first_number(config.get("layer_height")) or 0.2

    n = len(de)
    meta = {
        "filament_diameter": fil_d,
        "filament_diameter_source": source,
        "layer_height_config": cfg_h,
        "nozzle_diameter": _first_number(config.get("nozzle_diameter")),
    }
    if n == 0:
        z3 = np.zeros((0, 3))
        return ToolPaths(z3, z3.copy(), *(np.zeros(0) for _ in range(3)),
                         np.zeros(0, int), np.zeros(0, int), types, meta)

    volume = np.asarray(de) * math.pi * 0.25 * fil_d**2
    height = np.asarray(hgt, dtype=float)
    height = np.where(np.isnan(height), cfg_h, height)
    p0a, p1a = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    length = np.linalg.norm(p1a - p0a, axis=1)
    width = np.asarray(wid, dtype=float)
    # Where the slicer gave no width, invert the usual "rounded rectangle" road model
    # area = h * (w - h * (1 - pi/4)).
    derived = volume / length / height + height * (1 - math.pi / 4)
    width = np.where(np.isnan(width), derived, width)
    return ToolPaths(p0a, p1a, volume, width, height,
                     np.asarray(tid, dtype=int), np.asarray(lay, dtype=int), types, meta)


def _num(s: str, default: float) -> float:
    try:
        return float(s)
    except ValueError:
        return default


def _first_number(s) -> float | None:
    if not s:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def _args(tokens) -> dict[str, float]:
    out = {}
    for t in tokens:
        if len(t) > 1 and t[0].isalpha():
            try:
                out[t[0].upper()] = float(t[1:])
            except ValueError:
                pass
    return out


def _target(cur: float, val, relative: bool) -> float:
    if val is None:
        return cur
    return cur + val if relative else val


def _arc_points(x0, y0, z0, x1, y1, z1, args, cw: bool):
    if "I" not in args and "J" not in args:
        return [(x0, y0, z0), (x1, y1, z1)]  # radius form is not supported; treat as a chord
    cx, cy = x0 + args.get("I", 0.0), y0 + args.get("J", 0.0)
    r = math.hypot(x0 - cx, y0 - cy)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    sweep = a1 - a0
    if cw:
        if sweep >= 0:
            sweep -= 2 * math.pi
    else:
        if sweep <= 0:
            sweep += 2 * math.pi
    if math.hypot(x1 - x0, y1 - y0) < 1e-9:  # full circle
        sweep = -2 * math.pi if cw else 2 * math.pi
    n = max(2, math.ceil(abs(sweep) * r / _ARC_CHORD))
    pts = []
    for i in range(n + 1):
        f = i / n
        a = a0 + sweep * f
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z0 + (z1 - z0) * f))
    pts[-1] = (x1, y1, z1)
    return pts
