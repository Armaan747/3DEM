"""Write results as legacy binary VTK (opens in ParaView, PyVista, VisIt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .solve import Result
from .voxelize import VoxelModel


def write_vtk(path, voxels: VoxelModel, result: Result | None = None) -> Path:
    """Write ``voxels`` (and ``result`` if given) as a structured-points VTK file.

    Cell data: fill, orientation harmonics and, with a result, all stress fields
    (void cells are written as 0 - threshold on ``fill``).  Point data: displacement.
    """
    path = Path(path)
    nx, ny, nz = voxels.shape
    fields: dict[str, np.ndarray] = {"fill": voxels.fill}
    if result is not None:
        fields.update({k: np.nan_to_num(v, nan=0.0) for k, v in result.cell.items() if k != "fill"})

    def block(a: np.ndarray) -> bytes:
        # VTK wants x varying fastest; our arrays are indexed [x, y, z].
        return np.ascontiguousarray(a.transpose(2, 1, 0), dtype=">f4").tobytes()

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nthreedem result\nBINARY\nDATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n".encode())
        o, s = voxels.origin, voxels.size
        f.write(f"ORIGIN {o[0]} {o[1]} {o[2]}\nSPACING {s} {s} {s}\n".encode())
        f.write(f"CELL_DATA {nx * ny * nz}\n".encode())
        for name, arr in fields.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            f.write(block(arr))
            f.write(b"\n")
        if result is not None:
            f.write(f"POINT_DATA {(nx + 1) * (ny + 1) * (nz + 1)}\n".encode())
            f.write(b"VECTORS displacement float\n")
            d = np.ascontiguousarray(result.displacement.transpose(2, 1, 0, 3), dtype=">f4")
            f.write(d.tobytes())
            f.write(b"\n")
    return path
