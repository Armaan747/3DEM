import numpy as np
import pytest

from threedem.materials import PRESETS, Material, get_material, harmonics, stress_transform, strain_transform


def test_transforms_are_power_conjugate():
    for theta in (0.0, 0.3, 1.2, np.pi / 2):
        assert np.allclose(stress_transform(theta).T @ strain_transform(theta), np.eye(6))


def test_stiffness_at_zero_is_local():
    m = PRESETS["PLA"]
    assert np.allclose(m.stiffness(0.0), m.local_stiffness())


def test_quarter_turn_swaps_in_plane_axes():
    m = PRESETS["PLA"]
    C0, C90 = m.stiffness(0.0), m.stiffness(np.pi / 2)
    assert C90[0, 0] == pytest.approx(C0[1, 1])
    assert C90[1, 1] == pytest.approx(C0[0, 0])
    assert C90[2, 2] == pytest.approx(C0[2, 2])


def test_isotropic_material_is_rotation_invariant():
    m = PRESETS["ISOTROPIC"]
    assert np.allclose(m.stiffness(0.7), m.stiffness(0.0), rtol=1e-3, atol=1e-2)


def test_harmonic_basis_reproduces_any_angle():
    m = PRESETS["PETG"]
    basis = m.harmonic_basis()
    for theta in (0.1, 0.77, 2.5, -1.0):
        f = np.array([h for h in harmonics(theta)]).ravel()
        assert np.allclose(np.einsum("m,mij->ij", f, basis), m.stiffness(theta), atol=1e-8)


def test_get_material_variants():
    assert get_material("pla").name == "PLA"
    custom = get_material({"base": "PLA", "name": "mine", "E3": 1800})
    assert custom.E3 == 1800 and custom.E1 == PRESETS["PLA"].E1
    with pytest.raises(ValueError):
        get_material("unobtainium")


def test_non_positive_definite_rejected():
    bad = Material("bad", 3000, 3000, 3000, 1000, 1000, 1000, 0.9, 0.9, 0.9)
    with pytest.raises(ValueError):
        bad.local_stiffness()
