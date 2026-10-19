"""GUI geometry helpers (pure numpy: no Qt or VTK needed)."""

import numpy as np
import pytest

from threedem import Model, parse_gcode, voxelize
from threedem.gui.geometry import (boundary_quads, node_displacement, nodes_of, pick_region, quads_within,
                                   section_mask)
from threedem.synth import bar_gcode


def make_model(length=6.0, width=4.0, height=2.0, size=1.0):
    text, dims = bar_gcode(length, width, height)
    vox = voxelize(parse_gcode(text), size, origin=(0, 0, 0),
                   shape=(round(dims["length"] / size), round(dims["width"] / size), round(dims["height"] / size)))
    return Model(vox, "PLA"), dims


def test_boundary_quads_of_a_box():
    model, _ = make_model()  # 6 x 4 x 2 voxels
    q = boundary_quads(model)
    assert len(q) == 2 * (6 * 4 + 6 * 2 + 4 * 2)  # surface area in voxel faces
    # every quad's corners are distinct compact nodes, and normals are unit axis vectors
    assert all(len(set(row)) == 4 for row in q.nodes.tolist())
    assert np.allclose(np.linalg.norm(q.normals(), axis=1), 1)


def test_quad_normals_point_away_from_the_solid():
    model, _ = make_model()
    q = boundary_quads(model)
    centre = model.node_xyz.mean(axis=0)
    mid = model.node_xyz[q.nodes].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", mid - centre, q.normals()) > 0)


def test_pick_region_selects_a_face_and_its_nodes():
    model, _ = make_model()
    q = boundary_quads(model)
    top = np.flatnonzero((q.axis == 2) & (q.sign > 0))
    region = pick_region(q, int(top[0]))
    assert len(region) == 6 * 4
    nodes = nodes_of(q, region)
    assert len(nodes) == 7 * 5
    assert np.allclose(model.node_xyz[nodes][:, 2], 2.0)
    assert len(quads_within(q, nodes)) >= len(region)


def test_pick_region_stops_at_edges_unless_whole_plane():
    # an L: the +Z faces lie on two different planes, so picking one must not leak into the other
    text, _ = bar_gcode(6, 4, 2)
    vox = voxelize(parse_gcode(text), 1.0, origin=(0, 0, 0), shape=(6, 4, 2))
    vox.fill[3:, :, 1] = 0  # remove the upper layer over half of the footprint
    model = Model(vox, "PLA")
    q = boundary_quads(model)
    top = np.flatnonzero((q.axis == 2) & (q.sign > 0))
    planes = set(q.plane[top].tolist())
    assert planes == {1, 2}
    hi = int(top[q.plane[top] == 2][0])
    assert len(pick_region(q, hi)) == 3 * 4
    lo = int(top[q.plane[top] == 1][0])
    assert len(pick_region(q, lo)) == 3 * 4


def test_section_exposes_cut_faces():
    model, _ = make_model()
    full = boundary_quads(model)
    cut = boundary_quads(model, section_mask(model, 0, 3.0, keep_below=True))
    assert len(cut) < len(full) + 4 * 2  # fewer side faces, plus one new cap
    caps = np.flatnonzero((cut.axis == 0) & (cut.sign > 0) & (cut.plane == 3))
    assert len(caps) == 4 * 2


def test_node_displacement_gathers_by_grid_index():
    model, _ = make_model()
    grid = np.zeros((*model.node_shape, 3))
    grid[..., 2] = np.arange(model.node_shape[2])[None, None, :]  # uz = k
    d = node_displacement(model, grid)
    assert d.shape == (model.n_nodes, 3)
    assert np.allclose(d[:, 2], model.node_xyz[:, 2])  # unit voxels from origin 0
