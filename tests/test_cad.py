"""Tests for fanscan.cad (cadquery impeller model + STEP export)."""
import math

import numpy as np
import pytest

from fanscan import cad, geometry
from fanscan.params import DEFAULT_PARAMS

P = DEFAULT_PARAMS


@pytest.fixture(scope="module")
def body():
    return cad.build_impeller(P)


def _analytic_volume(p) -> float:
    """backplate + hub + Z * blade_len * t * b (SPEC estimate, hub/plate overlap ignored)."""
    r, theta = geometry.camber_line(p.r1, p.r2, p.blade_angle_inlet, p.blade_angle_outlet, n=256)
    x, y = geometry.polar_to_xy(r, theta)
    blade_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    tb = p.backplate_thickness
    hub_h = p.hub_height if p.hub_height is not None else p.blade_width + tb
    v_plate = math.pi * (p.r2 + 2.0) ** 2 * tb
    v_hub = math.pi * p.r_hub ** 2 * hub_h
    v_blades = p.blade_count * blade_len * p.blade_thickness * p.blade_width
    return v_plate + v_hub + v_blades


def test_build_single_valid_solid(body):
    assert len(body.solids().vals()) == 1
    assert body.val().isValid()


def test_bounding_box(body):
    bb = body.val().BoundingBox()
    tb = P.backplate_thickness
    assert bb.zmin == pytest.approx(0.0, abs=1e-3)
    assert bb.zmax == pytest.approx(tb + P.blade_width, abs=1e-3)
    r_plate = P.r2 + 2.0
    assert bb.xmax == pytest.approx(r_plate, abs=0.05)
    assert bb.xmin == pytest.approx(-r_plate, abs=0.05)
    assert bb.ymax == pytest.approx(r_plate, abs=0.05)
    assert bb.ymin == pytest.approx(-r_plate, abs=0.05)


def test_volume_matches_analytic(body):
    v = body.val().Volume()
    est = _analytic_volume(P)
    assert abs(v - est) / est < 0.10


def test_export_step(tmp_path):
    out = cad.export_step(P, tmp_path / "impeller.step")
    assert out.exists()
    assert out.stat().st_size > 10_000
    head = out.read_bytes()[:64]
    assert head.startswith(b"ISO-10303-21")


def test_export_stl(tmp_path):
    out = cad.export_stl(P, tmp_path / "impeller.stl")
    assert out.exists() and out.stat().st_size > 1_000
