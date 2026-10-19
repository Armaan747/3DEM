"""Verification of the full chain G-code -> voxels -> FE solution against closed-form results."""

import json

import numpy as np
import pytest

from threedem import Model, parse_gcode, run_case, voxelize
from threedem.materials import PRESETS
from threedem.synth import bar_gcode

MAT = PRESETS["PLA"]


def build(length, width, height, pattern, size, material=MAT, pitch=0.4):
    text, dims = bar_gcode(length, width, height, pitch=pitch, pattern=pattern)
    tp = parse_gcode(text)
    shape = tuple(round(d / size) for d in (dims["length"], dims["width"], dims["height"]))
    vox = voxelize(tp, size, origin=(0, 0, 0), shape=shape)
    return Model(vox, material), dims


def pin_axial(model, axis):
    """Support a bar for uniaxial loading along `axis` without restraining Poisson contraction."""
    p, q = [a for a in "xyz" if a != axis]
    model.fix(model.select_nodes(face=f"{axis}min"), axis)  # axial translation and both tilts
    model.fix(model.select_nodes(box={axis: [0, 0], p: [0, 0], q: [0, 0]}), p + q)  # origin corner
    p_max = model.node_xyz[:, "xyz".index(p)].max()
    # one more node, offset along p, stops the rotation about the load axis
    model.fix(model.select_nodes(box={axis: [0, 0], p: [p_max, p_max], q: [0, 0]}), q)


@pytest.mark.parametrize(
    "pattern,load_axis,modulus",
    [("x", "x", MAT.E1), ("y", "x", MAT.E2), ("x", "y", MAT.E2), ("x", "z", MAT.E3)],
)
def test_uniaxial_modulus_follows_raster_direction(pattern, load_axis, modulus):
    dims_in = {"x": 8.0, "y": 4.0, "z": 4.0}
    model, dims = build(dims_in["x"], dims_in["y"], dims_in["z"], pattern, size=0.5)
    pin_axial(model, load_axis)
    force = 100.0
    vec = [0.0, 0.0, 0.0]
    vec["xyz".index(load_axis)] = force
    model.add_force(model.select_nodes(face=f"{load_axis}max"), vec)
    res = model.solve()

    length = dims[{"x": "length", "y": "width", "z": "height"}[load_axis]]
    area = np.prod([dims[k] for k in ("length", "width", "height")]) / length
    top = res.displacement[..., "xyz".index(load_axis)]
    # mean displacement of the loaded face, relative to the supported face
    ax = "xyz".index(load_axis)
    sl_hi = [slice(None)] * 3
    sl_hi[ax] = -1
    sl_lo = [slice(None)] * 3
    sl_lo[ax] = 0
    delta = top[tuple(sl_hi)].mean() - top[tuple(sl_lo)].mean()
    measured = force * length / (area * delta)
    assert measured == pytest.approx(modulus, rel=0.02)
    np.testing.assert_allclose(res.reaction + res.applied, 0, atol=1e-6 * force)


def test_cantilever_matches_timoshenko_beam():
    L, W, H = 30.0, 6.0, 6.0
    model, dims = build(L, W, H, "x", size=0.5)
    model.fix(model.select_nodes(face="xmin"), "xyz")
    P = 10.0
    model.add_force(model.select_nodes(face="xmax"), [0, 0, -P])
    res = model.solve()

    I = W * H**3 / 12
    bending = P * L**3 / (3 * MAT.E1 * I)
    shear = P * L / ((5 / 6) * MAT.G13 * W * H)
    expected = bending + shear
    tip = -res.displacement[-1, :, :, 2].mean()
    # A clamped end that also blocks Poisson contraction and trilinear-element shear
    # locking both stiffen the model slightly, so the numerical answer is a bit low.
    assert tip == pytest.approx(expected, rel=0.08)
    assert tip < expected


def test_interlayer_weakness_shows_up_in_safety_factor():
    # Same cube, same load (hence same stress), pulled along Z vs along X: Z must be flagged worse.
    sf = {}
    for axis in ("x", "z"):
        model, dims = build(4.0, 4.0, 4.0, "x", size=0.5)
        pin_axial(model, axis)
        vec = [0.0, 0.0, 0.0]
        vec["xyz".index(axis)] = 100.0
        model.add_force(model.select_nodes(face=f"{axis}max"), vec)
        sf[axis] = np.nanmin(model.solve().cell["safety_factor"])
    assert sf["z"] < sf["x"]


def test_floating_island_is_removed_with_warning():
    a, _ = bar_gcode(6, 4, 2, origin=(0, 0))
    b, _ = bar_gcode(6, 4, 2, origin=(20, 0))
    tp = parse_gcode(a + b)
    vox = voxelize(tp, 0.5, origin=(0, 0, 0), shape=(52, 8, 4))
    model = Model(vox, MAT)
    model.fix(model.select_nodes(box={"x": [0, 0]}), "xyz")
    model.add_force(model.select_nodes(box={"x": [6, 6]}), [10, 0, 0])
    with pytest.warns(UserWarning, match="not connected"):
        res = model.solve()
    assert res.info["floating_elements_removed"] > 0


def test_no_supports_is_an_error():
    model, _ = build(4, 2, 1, "x", size=0.5)
    with pytest.raises(ValueError, match="supports"):
        model.solve()


def test_case_file_end_to_end(tmp_path):
    text, _ = bar_gcode(10, 4, 4, pattern="alt")
    (tmp_path / "bar.gcode").write_text(text)
    case = {
        "gcode": "bar.gcode",
        "voxel_size": 0.5,
        "material": "PETG",
        "supports": [{"face": "xmin"}],
        "loads": [{"face": "xmax", "force": [0, 0, -5]}],
        "output": "out.vtk",
    }
    (tmp_path / "case.json").write_text(json.dumps(case))
    vox, model, res = run_case(case, tmp_path, verbose=False)
    assert (tmp_path / "out.vtk").stat().st_size > 1000
    assert res.max_displacement() > 0
    assert res.applied[2] == pytest.approx(-5.0)
