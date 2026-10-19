"""threedem - finite element analysis of 3D-printed parts, driven by the slicer's G-code."""

from .case import run_case
from .gcode import DEFAULT_EXCLUDE, ToolPaths, parse_gcode
from .materials import PRESETS, Material, get_material
from .solve import Model, Result
from .voxelize import VoxelModel, voxelize

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_EXCLUDE", "Material", "Model", "PRESETS", "Result", "ToolPaths", "VoxelModel",
    "get_material", "parse_gcode", "run_case", "voxelize",
]
