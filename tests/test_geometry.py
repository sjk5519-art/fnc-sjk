"""Tests for fanscan.geometry (camber-line model and its inverse)."""
import math

import numpy as np
import pytest

from fanscan import geometry
from fanscan.params import DEFAULT_PARAMS

P = DEFAULT_PARAMS


@pytest.mark.parametrize("beta1,beta2", [
    (P.blade_angle_inlet, P.blade_angle_outlet),  # default 32 -> 45
    (20.0, 20.0),                                 # constant beta (log spiral)
    (32.0, 90.0),                                 # radial outlet
    (60.0, 30.0),                                 # beta decreasing outward
])
def test_camber_line_round_trip(beta1, beta2):
    r, theta = geometry.camber_line(P.r1, P.r2, beta1, beta2, n=256)
    assert theta[0] == 0.0
    assert np.all(np.diff(theta) > 0), "blade must sweep toward +theta as r grows"
    b1, b2 = geometry.fit_linear_beta(r, theta, P.r1, P.r2)
    assert abs(b1 - beta1) < 0.1
    assert abs(b2 - beta2) < 0.1


def test_camber_line_sweep_consistent():
    r, theta = geometry.camber_line(P.r1, P.r2, P.blade_angle_inlet, P.blade_angle_outlet)
    sweep = geometry.total_sweep_deg(P.r1, P.r2, P.blade_angle_inlet, P.blade_angle_outlet)
    assert sweep > 0
    assert math.degrees(theta[-1]) == pytest.approx(sweep, abs=0.05)
    assert r[0] == P.r1 and r[-1] == P.r2


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Proper intersection test of segments p1p2 and p3p4 (shared endpoints excluded)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def test_offset_camber_polygon_closed_and_simple():
    r, theta = geometry.camber_line(P.r1, P.r2, P.blade_angle_inlet, P.blade_angle_outlet, n=64)
    poly = geometry.offset_camber(r, theta, P.blade_thickness)

    assert poly.ndim == 2 and poly.shape[1] == 2
    assert poly.shape[0] == 2 * len(r)
    assert np.all(np.isfinite(poly))

    # The polygon is implicitly closed: the closing edge joins the last and
    # first vertex, both on the same (inlet) flat end, |edge| == thickness.
    closing = np.linalg.norm(poly[-1] - poly[0])
    assert closing == pytest.approx(P.blade_thickness, rel=1e-6)
    # The two offset sides are `thickness` apart at every station.
    n = len(r)
    left, right = poly[:n], poly[n:][::-1]
    assert np.allclose(np.linalg.norm(left - right, axis=1), P.blade_thickness, rtol=1e-6)

    # Non-zero area (shoelace) close to length * thickness.
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    cx, cy = geometry.polar_to_xy(r, theta)
    length = np.sum(np.hypot(np.diff(cx), np.diff(cy)))
    assert area == pytest.approx(length * P.blade_thickness, rel=0.02)

    # No self-intersection: check every pair of non-adjacent edges (closing edge included).
    pts = [tuple(p) for p in poly]
    m = len(pts)
    edges = [(pts[i], pts[(i + 1) % m]) for i in range(m)]
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue  # adjacent via the closing edge
            assert not _segments_intersect(*edges[i], *edges[j]), f"edges {i} and {j} cross"


def test_beta_from_trace_sign_robust():
    r, theta = geometry.camber_line(P.r1, P.r2, 32.0, 45.0, n=128)
    b_pos = geometry.fit_linear_beta(r, theta, P.r1, P.r2)
    b_neg = geometry.fit_linear_beta(r, -theta, P.r1, P.r2)
    assert b_pos == pytest.approx(b_neg, abs=1e-9)
