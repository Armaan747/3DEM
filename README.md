# threedem

Finite element analysis of 3D-printed parts, driven by the **G-code your slicer actually produced**.

Slice in OrcaSlicer (or PrusaSlicer, Bambu Studio, Cura), export the G-code, then apply supports
and forces and see displacements, stresses and a safety factor. Because the model is built from the
toolpaths, it knows about your walls, infill pattern and density, raster directions and layer
bonds - not just the outer shape.

> **Status: early alpha (v0.1).** The numerics are verified against closed-form solutions (see
> [Verification](#verification)), but nothing has been validated against physical print tests yet,
> and the material numbers are illustrative. Do not use it for safety-critical decisions.

## How it works

```
G-code ─► toolpath segments ─► voxel grid ─► orthotropic hex8 FE model ─► stresses
          (position, volume,   (fill fraction,  (supports + loads you
           width, layer, type)  raster direction)  define)
```

1. **Parse** the G-code: linear and arc moves, relative/absolute extrusion, retraction, `;TYPE:`
   feature comments, and the slicer's config block (filament diameter).
2. **Voxelize**: each segment deposits its extruded volume onto the grid (volume is conserved
   exactly) together with its raster direction.
3. **Material**: printed material is orthotropic - stiff along the line, weaker across lines, weakest
   across layers. Rotated per voxel from the raster direction; mixed directions in one voxel are averaged.
4. **Solve** linear static elasticity with 8-node hexahedra (sparse direct solver, or CG for large models).
5. **Report** von Mises stress in the material, in-plane and inter-layer failure indices, a safety
   factor, and displacements, written to a VTK file for ParaView/PyVista.

## Install

```bash
git clone <this repo>
cd threedem
pip install -e ".[dev]"      # add ,fast for the pyamg preconditioner
pytest
```

## Try it

```bash
threedem demo                      # writes + solves a small cantilever in ./threedem_demo
threedem info my_part.gcode        # bounds, layers, plastic volume by feature type
threedem run my_case.json          # run an analysis
```

Open the resulting `.vtk` in [ParaView](https://www.paraview.org/) and threshold on `fill` to hide void.

A real example is in [`tests/Geom`](tests/Geom): an L bracket sliced with OrcaSlicer 2.3.1 (3 walls, 10 % rectilinear
infill), with its CAD/STL and a ready-to-run case that bolts down the far end of the base and pushes the top of the upright:

```bash
threedem run tests/Geom/l_bracket_case.json
```

### Case file

```json
{
  "gcode": "bracket.gcode",
  "voxel_size": 0.5,
  "material": "PLA",
  "exclude_types": ["skirt", "brim", "support", "custom"],
  "supports": [{"face": "zmin", "axes": "xyz"}],
  "loads":    [{"box": {"z": [19.5, null]}, "force": [0, 0, -50]}],
  "output": "bracket_result.vtk"
}
```

* Coordinates are **G-code (bed) coordinates in mm** - the same ones you see in the slicer.
  `threedem info` prints the part's bounds.
* A selection is a `box` (`{"x": [lo, hi], ...}`, `null` = open) and/or a `face` (`xmin`, `xmax`, `ymin`,
  `ymax`, `zmin`, `zmax`, optionally with a `depth` in mm).
* `supports` fix the listed `axes` of the selected nodes. `loads` apply a total force in N, distributed
  over the selection by tributary area.
* `material` is a preset (`PLA`, `PETG`, `ABS`, `ISOTROPIC`) or an object overriding fields of a preset:
  `{"base": "PLA", "E3": 1800, "strength_3": 20}`. Units: N, mm, MPa.

### From Python

```python
from threedem import parse_gcode, voxelize, Model

paths = parse_gcode("part.gcode").exclude()
vox = voxelize(paths, size=0.5)
model = Model(vox, "PETG")
model.fix(model.select_nodes(face="zmin"))
model.add_force(model.select_nodes(box={"z": [19.5, None]}), [0, 0, -50])
result = model.solve()
print(result.summary())
```

## Verification

The test-suite checks the whole chain (G-code → voxels → solution) against closed-form results:

* uniaxial tension along X, Y and Z recovers `E1`, `E2` and `E3` of the material to within 2%, depending only
  on the raster direction in the G-code;
* a cantilever's tip deflection matches Timoshenko beam theory within 3% (hex8 elements are slightly stiff in bending);
* plastic volume is conserved from G-code to voxel grid;
* reactions balance applied loads; floating islands are detected and removed.

## Known limitations (read before trusting numbers)

* **Material data are illustrative defaults, not measurements.** Print and break your own coupons
  (X, Y and Z orientation) and enter the results. Calibration is the biggest source of error.
* **Resolution matters.** Bending of thin walls needs several voxels through the thickness; use a
  voxel size near the road width (0.4-0.5 mm) for walls and small parts. Coarse voxels give a
  homogenized, cheaper, but stiffer-than-real answer for thin features.
* **Linear elastic only.** No plasticity, creep, contact, buckling or large deformation. Real PLA/PETG
  creeps and yields - treat the safety factor as a preliminary screening index.
* The failure indices are simple (in-plane von Mises vs. the weaker in-plane strength; inter-layer tension +
  shear). Tsai-Wu/Hashin style criteria are on the roadmap.
* Hex8 elements show shear locking in thin bending; incompatible-mode or higher-order elements are planned.
* ASCII G-code only (no `.bgcode`), and no viewer yet - results go to VTK.
* Z-hop/travel, wipes and retractions are handled, but exotic firmware flavours and non-planar slicing are untested.
  Feedback with sample files is welcome.

## Roadmap

1. Interactive viewer for choosing supports and loads by clicking (PyVista/Qt)
2. Matrix-free multigrid solver for fine meshes, optional GPU
3. Incompatible-mode hex8 to remove bending locking
4. Calibrated material library and coupon-fitting helper
5. Better strength criteria (Tsai-Wu, interlayer cohesive limits)
6. Load the slicer's `.3mf` to pick faces on the original mesh
7. Validation against printed and tested parts

## License

MIT - see [LICENSE](LICENSE). OrcaSlicer is AGPL-3.0; this project only reads the G-code it produces and
contains no Orca source code.
