"""Regression tests for the eight analyzer defects fixed in analyze.py / synth.py.

Each test family mirrors one scenario family of the fix agent's regression
script (23 scenarios, all within tolerance after the fix).  Tolerances are the
SPEC ones (D within 3 %, beta within 4 deg, Z exact, thickness within 30 %)
except where a measured, documented resolution caveat is noted inline.  The
"measured" numbers in comments are the post-fix errors; before the fix the
same scenarios were off by up to 41 % on D2, mis-counted Z by 2-3x, or raised.
"""
import dataclasses
import functools

import cv2
import numpy as np
import pytest

from fanscan import analyze, synth
from fanscan.params import DEFAULT_PARAMS

REF_D = 30.0
TOL_D = 0.03       # SPEC: diameters within 3 %
TOL_BETA = 4.0     # SPEC: blade angles within 4 deg
TOL_T = 0.30       # SPEC: thickness within 30 %


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _params(**over):
    p = dataclasses.replace(DEFAULT_PARAMS, meta={}, **over)
    p.validate()
    return p


@functools.lru_cache(maxsize=None)
def _render(over: tuple, render_kw: tuple):
    """Module-level cache so families that share an image render it once."""
    p = _params(**dict(over))
    img, gt = synth.render(p, ref_diameter_mm=REF_D, **dict(render_kw))
    img.setflags(write=False)
    return p, img, gt


def _case(over: dict, render_kw: dict):
    return _render(tuple(sorted(over.items())), tuple(sorted(render_kw.items())))


def _rel(est, truth):
    return abs(est - truth) / truth


def _assert_diameters(est, truth, tol):
    for name in ("outer_diameter", "inner_diameter", "hub_diameter"):
        t, e = getattr(truth, name), getattr(est, name)
        assert _rel(e, t) <= tol, f"{name}: truth {t} est {e} ({100 * (e / t - 1):+.1f} %)"


def _assert_angles(est, truth):
    assert abs(est.blade_angle_inlet - truth.blade_angle_inlet) <= TOL_BETA, \
        f"beta1: truth {truth.blade_angle_inlet} est {est.blade_angle_inlet}"
    assert abs(est.blade_angle_outlet - truth.blade_angle_outlet) <= TOL_BETA, \
        f"beta2: truth {truth.blade_angle_outlet} est {est.blade_angle_outlet}"


# ---------------------------------------------------------------------------
# 1. backplate disc fit with thick blades / low resolution
#    Before the fix the disc fit latched onto blade tips (D2 -8 %, centre off by
#    up to 16 px, hub -11 %).  Measured after the fix: D2/D1 within 0.9 % at
#    px >= 1.5, centre within 0.05 px.
# ---------------------------------------------------------------------------
DISC_CASES = [
    # id,            params override,            render kwargs,          D tolerance
    ("t5-px4",       dict(blade_thickness=5.0),   dict(px_per_mm=4.0),    TOL_D),
    ("t4-px4",       dict(blade_thickness=4.0),   dict(px_per_mm=4.0),    TOL_D),
    ("t3-px2",       dict(blade_thickness=3.0),   dict(px_per_mm=2.0),    TOL_D),
    ("t8-px4",       dict(blade_thickness=8.0),   dict(px_per_mm=4.0),    TOL_D),
    # px = 1.0 mm/px: the 2 mm blades are 2 px wide and the 1 mm rim is one pixel,
    # so edge localisation is limited by sampling, not by the fit.  Measured
    # D2 -1.4 % / D1 -1.5 % after the fix (vs -10.6 % / -4.8 % before); the
    # documented caveat in the regression script is 4 % at this resolution.
    ("default-px1",  {},                          dict(px_per_mm=1.0),    0.04),
    ("Z16-px1.5",    dict(blade_count=16),        dict(px_per_mm=1.5),    TOL_D),
    ("Z9-px1.5",     {},                          dict(px_per_mm=1.5),    TOL_D),
]


@pytest.mark.parametrize("over,render_kw,tol_d", [c[1:] for c in DISC_CASES],
                         ids=[c[0] for c in DISC_CASES])
def test_disc_fit_thick_blades_low_resolution(over, render_kw, tol_d):
    truth, img, gt = _case(over, render_kw)
    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    est = res.params

    assert est.blade_count == truth.blade_count
    _assert_diameters(est, truth, tol_d)
    _assert_angles(est, truth)
    cx, cy = res.center_xy
    tcx, tcy = gt["center_xy"]
    assert abs(cx - tcx) <= 1.0 and abs(cy - tcy) <= 1.0, \
        f"centre off by ({cx - tcx:+.2f}, {cy - tcy:+.2f}) px"
    assert res.px_per_mm == pytest.approx(gt["px_per_mm"], rel=TOL_D)


# ---------------------------------------------------------------------------
# 2. blade count on few / many blades under noise
#    Before the fix the comb score picked harmonics: Z=2 -> 4 or 6, Z=3 -> 6.
#    The fix agent's 120-trial grid (Z x seed x blur x rot, noise 8) passes.
# ---------------------------------------------------------------------------
Z_CASES = [(Z, seed) for Z in (2, 3, 4, 5, 24, 32) for seed in (0, 2)]


@pytest.mark.parametrize("Z,seed", Z_CASES, ids=[f"Z{z}-seed{s}" for z, s in Z_CASES])
def test_blade_count_noisy(Z, seed):
    truth, img, _ = _case(dict(blade_count=Z),
                          dict(noise_sigma=8.0, blur_sigma=1.5, rotation_deg=13.0, seed=seed))
    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    assert res.params.blade_count == Z, \
        f"Z {res.params.blade_count} != {Z}; votes={res.debug.get('blade_count_votes')}"
    # the geometry must still come out right once Z is correct
    _assert_diameters(res.params, truth, TOL_D)
    _assert_angles(res.params, truth)


# ---------------------------------------------------------------------------
# 3. input dtypes
#    Before the fix _to_gray clipped uint16 / float inputs to [0, 255], which
#    saturated the image and raised "reference circle not found".
# ---------------------------------------------------------------------------
def _as_uint16(img):
    return img.astype(np.uint16) * 257          # what cv2.imread(IMREAD_UNCHANGED) yields for 16-bit PNG


def _as_float01(img):
    return img.astype(np.float32) / 255.0


def _as_bgr(img):
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)


DTYPE_CASES = [("uint16", _as_uint16), ("float32-01", _as_float01), ("bgr", _as_bgr)]


@pytest.fixture(scope="module")
def uint8_reference():
    _truth, img, _gt = _case({}, dict(px_per_mm=4.0))
    return img, analyze.analyze(img, ref_diameter_mm=REF_D)


@pytest.mark.parametrize("convert", [c[1] for c in DTYPE_CASES], ids=[c[0] for c in DTYPE_CASES])
def test_input_dtypes_match_uint8(uint8_reference, convert):
    img8, ref = uint8_reference
    res = analyze.analyze(convert(img8), ref_diameter_mm=REF_D)
    assert res.params.blade_count == ref.params.blade_count
    for name in ("outer_diameter", "inner_diameter", "hub_diameter",
                 "blade_angle_inlet", "blade_angle_outlet", "blade_thickness"):
        e, r = getattr(res.params, name), getattr(ref.params, name)
        assert _rel(e, r) <= 0.005, f"{name}: uint8 {r} vs converted {e}"
    assert res.px_per_mm == pytest.approx(ref.px_per_mm, rel=0.005)
    assert res.center_xy == pytest.approx(ref.center_xy, abs=0.5)


# ---------------------------------------------------------------------------
# 4. long blades (small D1): the blade band must not be cut off at the inner
#    edge.  Before the fix D2 came out -41 % / -23 % / -13 % for these.
#    Measured after the fix: D2 within 0.3 %.
# ---------------------------------------------------------------------------
LONG_CASES = [
    ("b10-b20-D1_40-hub12", dict(blade_angle_inlet=10.0, blade_angle_outlet=20.0,
                                 inner_diameter=40.0, hub_diameter=12.0)),
    ("D1_30-hub10",         dict(inner_diameter=30.0, hub_diameter=10.0)),
]


@pytest.mark.parametrize("over", [c[1] for c in LONG_CASES], ids=[c[0] for c in LONG_CASES])
def test_long_blades_outer_diameter(over):
    truth, img, _ = _case(over, dict(px_per_mm=4.0))
    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    est = res.params
    assert est.blade_count == truth.blade_count
    assert _rel(est.outer_diameter, truth.outer_diameter) <= TOL_D, \
        f"D2: truth {truth.outer_diameter} est {est.outer_diameter}"
    assert _rel(est.hub_diameter, truth.hub_diameter) <= TOL_D
    _assert_angles(est, truth)


def test_overlapping_blades_are_flagged_not_guessed():
    """Z=16 with D1=30 and shallow angles: blades cover more than half the
    circumference near the eye, so D1 is not measurable (measured -5 %).  The
    analyzer must say so instead of reporting a confident number; D1 accuracy
    is deliberately *not* asserted here."""
    truth, img, _ = _case(dict(blade_count=16, inner_diameter=30.0, hub_diameter=10.0,
                               blade_angle_inlet=20.0, blade_angle_outlet=25.0),
                          dict(px_per_mm=4.0))
    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    warnings = res.params.meta["warnings"]
    assert any("blades cover more than half" in w for w in warnings), warnings
    assert res.confidence["inner_diameter"] <= 0.3
    # everything that *is* observable still has to be right
    assert res.params.blade_count == 16
    assert _rel(res.params.outer_diameter, truth.outer_diameter) <= TOL_D
    _assert_angles(res.params, truth)


# ---------------------------------------------------------------------------
# 5. thin blade thickness
#    Before the fix the ridge-width walk over-estimated t by +31 % (t=1, px4)
#    and +37 % (t=2, px1.5).  Measured after the fix: -4 % and -5 %.
# ---------------------------------------------------------------------------
THIN_CASES = [("t1-px4", dict(blade_thickness=1.0), dict(px_per_mm=4.0)),
              ("t2-px1.5", dict(blade_thickness=2.0), dict(px_per_mm=1.5))]


@pytest.mark.parametrize("over,render_kw", [c[1:] for c in THIN_CASES], ids=[c[0] for c in THIN_CASES])
def test_thin_blade_thickness(over, render_kw):
    truth, img, _ = _case(over, render_kw)
    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    t, e = truth.blade_thickness, res.params.blade_thickness
    assert _rel(e, t) <= TOL_T, f"thickness: truth {t} est {e} ({100 * (e / t - 1):+.0f} %)"
    assert res.params.blade_count == truth.blade_count


# ---------------------------------------------------------------------------
# 6. seed search in _trace_blades must visit every column of [c_lo, c_hi]
#    The old loop `range(0, (c_hi - c_lo) // 2 + 1)` skipped c_hi on odd spans,
#    so a blade set only resolvable at the outermost radius raised
#    "could not find Z blade minima at any radius".
# ---------------------------------------------------------------------------
def _polar_with_sole_seed(Z, H, W, good_col, offset=30.0, width=4.0):
    """Flat polar image whose only column with Z minima is *good_col*."""
    pol = np.full((H, W), 128.0)
    th = np.arange(H)[:, None]
    centers = (np.arange(Z) * H / Z + offset)[None, :]
    d = (th - centers + H / 2) % H - H / 2
    pol[:, good_col] = 128 - 60 * np.exp(-0.5 * (d / width) ** 2).sum(1)
    return pol


SEED_SPANS = [(10, 17), (0, 5), (3, 4)]      # the reviewer's odd spans


@pytest.mark.parametrize("c_lo,c_hi", SEED_SPANS, ids=[f"span{lo}-{hi}" for lo, hi in SEED_SPANS])
def test_trace_seed_search_reaches_c_hi(c_lo, c_hi):
    Z, H = 6, 360
    pol = _polar_with_sole_seed(Z, H, c_hi + 5, c_hi)
    dbg = {}
    traces = analyze._trace_blades(pol, Z, c_lo, c_hi, 2 * np.pi / H, dbg)
    assert dbg["trace_seed_col"] == c_hi
    assert len(traces) == Z


@pytest.mark.parametrize("c_lo,c_hi", SEED_SPANS, ids=[f"span{lo}-{hi}" for lo, hi in SEED_SPANS])
def test_trace_seed_search_visits_every_column(c_lo, c_hi):
    Z, H = 6, 360
    for good in range(c_lo, c_hi + 1):
        pol = _polar_with_sole_seed(Z, H, c_hi + 3, good)
        dbg = {}
        analyze._trace_blades(pol, Z, c_lo, c_hi, 2 * np.pi / H, dbg)
        assert dbg["trace_seed_col"] == good, f"sole seed at column {good} not chosen"
