"""End-to-end tests of the shrouded-mode analyzer (fanscan.shrouded) on render3d images and on
the real phone photo, plus the ``analyze --mode shrouded`` CLI path.

The asserts below reflect what the analyzer *as built* achieves (measured 2026-11-02 with the
analyzer author's grids and re-measured by this file's runs), not the SPEC2 wish list.  Where
the analyzer returns a prior by design, the provenance / confidence contract is asserted
instead of accuracy.  Nothing is skipped silently: cases the analyzer is known to get wrong
are ``xfail(strict=True)`` tests, so an improvement flips them loudly.

GRID PARAMS ("realistic" variant, what the real sample looks like): SHROUDED_PARAMS_FACTORY()
with inner_diameter = 120 (LE at r1 = 60 well inside the eye, r_eye = 90) and a short hub
flange (hub_height = 20).  The factory's own r1 = 95 > r_eye = 90 hides every blade root in
an 'axial_far' render (truth visible_blade_fraction = 0.00), so Z is a prior there *by
construction* - that contract is tested in test_factory_axial_far_blade_count_is_prior.
Renders are 1200 px, seed = 7 Z + beta2 (the author's grid seeds), D2 and D_eye given, so the
perspective correction runs.

ACHIEVED (worst case over the 9 (Z, beta2) combinations per preset, clean renders) -> assert:

    quantity                          axial_far      axial_close    tilt10         assert
    ---------------------------------------------------------------------------------------
    Z exact ('measured')              9/9 (conf.99)  9/9 (conf.98)  9/9 (conf.75)  exact, conf >= 0.7
    D_eye image-measured, persp.-     1.3 %          6.4 % (two     3.2 %          3 % / 7 % / 4 %
      corrected (reported value is                   clusters: 0.0
      'user' = exact)                                and 6.4 %)
    hub                               'prior' 0.3 D2 (= truth 90 by coincidence) in every axial
                                      case; 'measured' within 0.3 % in 6/9 tilt10 cases -> contract
    D1 (r1 = innermost traced r)      3.4 %          5.3 %          3.5 %          4 % / 6 % / 4 %
    beta1 'measured'                  1.0 deg        1.2 deg        1.4 deg        2.0 deg
    beta2 'extrapolated'              2.3 deg        1.1 deg        5.8 deg        3 / 3 / 7 deg
    beta2 confidence                  0.35           0.55-0.61      0.09-0.26      close >= 0.4
    camera distance (meta)            -16 %          +2.2 %         -41 %          close within 5 %
    analyze_shrouded runtime          2.1 s          2.9 s          4.7 s          < 8 s

CLUTTER (9-case subset, one per (Z, beta2), presets cycling): Z 'measured' and exact in 8/9,
'prior' (conf 0.05) in 1/9 (Z9 b55 tilt10) -> Z asserted only when 'measured', else conf <= 0.1,
and >= 2/3 of the subset must be 'measured'.  beta1 within 4.3 deg when measured (assert 5).
Over the full 27-case clutter grid the analyzer MIS-COUNTS once: Z9 b40 axial_far -> Z = 5
'measured' with confidence 0.69 (test_clutter_known_miscount, xfail strict).

WRAP-ANGLE beta2 (axial_close with a bright marker + 'side' views at azimuths 240/270/300,
D2 + D_eye given): (Z11, b40) -> 40.36 deg, (Z9, b55) -> 55.67 deg, both 'wrap' with
confidence 0.70-0.73 -> assert within 3 deg, conf >= 0.5; blade width from the rim gap
'measured' within 12 % (assert 15 %).  (Z13, b30) -> 26.55 deg (wrap 99.9 vs truth 90.9):
3.45 deg off, test_wrap_angle_known_bias (xfail strict).

SCALE FALLBACKS (Z11 b40 axial_far): no scale -> D2 300 'prior' conf 0.05, eye/D2 ratio
within 0.9 %; D_eye only -> D2 'measured' within 0.9 %.  D2 only at axial_close: the eye ring
sits ~35 mm nearer the camera and is reported +13.1 % too large with a 'far camera assumed'
warning (documented limitation, test_d2_only_close_camera_bias).

REAL PHOTO samples/real_axial_eye.jpg (no truth; 8 blades by eye): rim + eye found, Z = 8
'measured' (conf 0.72), beta1 30.5 deg (conf 0.26), beta2 33.8 'extrapolated' (conf 0.07),
blades sweep clockwise, 6.4-7.1 s.

RUNTIME (4-vCPU dev box, 2026-11-02): 45 passed, 2 xfailed.  Cold run 358 s (10 meshes built
in 134 s + 49 renders in 222 s, both cached on disk by the ``render_cache`` fixture in
tests/conftest.py); warm run 141 s.  Run with ``-s`` to get the module summary (worst
errors per preset of this run; use it to refresh the table above).  Note for the clutter
subset: in the (Z9 b40 axial_close) and (Z13 b40 tilt10) cases the rim ellipse is picked
~175 px off the renderer's rim (a clutter edge), yet Z is still measured correctly; the rim
position is therefore asserted against the truth only in the clean axial cases.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from fanscan import geometry, shrouded
from fanscan.params import SHROUDED_PARAMS_FACTORY, ImpellerParams

ROOT = Path(__file__).resolve().parents[1]
REAL_AXIAL = ROOT / "samples" / "real_axial_eye.jpg"
PROVENANCE_SOURCES = {"measured", "extrapolated", "wrap", "prior", "user"}

# ---- tolerances (achieved numbers in the header) -------------------------------------------
Z_CONF_MIN = 0.7
D_EYE_TOL = {"axial_far": 0.03, "axial_close": 0.07, "tilt10": 0.04}   # image-measured D_eye
D1_TOL = {"axial_far": 0.04, "axial_close": 0.06, "tilt10": 0.04}
HUB_MEASURED_TOL = 0.04
BETA1_TOL_DEG = 2.0
BETA1_CLUTTER_TOL_DEG = 5.0
BETA2_EXTRAP_TOL_DEG = {"axial_far": 3.0, "axial_close": 3.0, "tilt10": 7.0}
BETA2_CLOSE_CONF_MIN = 0.4
BETA2_WRAP_TOL_DEG = 3.0
BETA2_WRAP_CONF_MIN = 0.5
B2_RIM_GAP_TOL = 0.15
CAM_DIST_CLOSE_TOL = 0.05
RIM_PX_TOL = 6.0
ANALYZE_TIME_BOUND_S = 8.0
REAL_TIME_BOUND_S = 15.0
CLUTTER_MEASURED_MIN_FRACTION = 2.0 / 3.0

GRID_SIZE = 1200
Z_VALUES = (9, 11, 13)
BETA2_VALUES = (30.0, 40.0, 55.0)
PRESET_SPECS = {"axial_far": ("axial_far", {}), "axial_close": ("axial_close", {}),
                "tilt10": ("tilt", {"tilt_deg": 10.0})}
CLEAN_GRID = list(itertools.product(Z_VALUES, BETA2_VALUES, PRESET_SPECS))
_preset_cycle = itertools.cycle(PRESET_SPECS)
CLUTTER_GRID = [(z, b, next(_preset_cycle)) for z, b in itertools.product(Z_VALUES, BETA2_VALUES)]
CLUTTER_KNOWN_MISCOUNT = (9, 40.0, "axial_far")      # Z = 5 'measured' conf 0.69
assert CLUTTER_KNOWN_MISCOUNT not in CLUTTER_GRID
WRAP_CASES = [(11, 40.0), (9, 55.0)]
WRAP_KNOWN_BIAS = (13, 30.0)                         # beta2 wrap 26.55 vs 30
SIDE_AZIMUTHS = (240.0, 270.0, 300.0)


def _ids(grid):
    return [f"Z{z}-b2_{b:g}-{p}" for z, b, p in grid]


# ---------------------------------------------------------------------------
# params / render helpers
# ---------------------------------------------------------------------------
def realistic_params(Z: int, beta2: float) -> ImpellerParams:
    p = dataclasses.replace(SHROUDED_PARAMS_FACTORY(), blade_count=Z, blade_angle_outlet=beta2,
                            inner_diameter=120.0, hub_height=20.0, meta={})
    p.validate()
    return p


def _render(render_cache, p: ImpellerParams, preset_name: str, *, clutter=False, seed=None, **kw):
    preset, extra = PRESET_SPECS.get(preset_name, (preset_name, {}))
    seed = (7 * p.blade_count + int(p.blade_angle_outlet)) if seed is None else seed
    return render_cache.render(p, preset, size=(GRID_SIZE, GRID_SIZE), clutter=clutter, seed=seed,
                               **extra, **kw)


def _grid_render(render_cache, Z, beta2, preset_name, clutter=False):
    p = realistic_params(Z, beta2)
    img, truth = _render(render_cache, p, preset_name, clutter=clutter)
    return img, truth, p


# ---------------------------------------------------------------------------
# result helpers
# ---------------------------------------------------------------------------
def _prov(res, name):
    p = res.provenance[name]
    return p.value, p.source, float(p.confidence)


def _eye_measured_mm(res) -> float:
    """Perspective-corrected eye diameter as measured in the image (the reported eye_diameter
    is 'user' = exact when D_eye is given)."""
    e = res.debug["eye_rect"]
    return 2.0 * math.sqrt(e.a * e.b) / res.debug["scale"]["s_eye"]


def _cam_dist_truth(truth) -> float:
    """Camera height above the shroud rim top edge (what the analyzer's cam_dist refers to)."""
    return float(truth["camera"]["position"][2]) - float(truth["z_levels"]["z_rim_top"])


def _check_result_structure(res, img):
    assert isinstance(res.params, ImpellerParams)
    res.params.validate()
    assert isinstance(res.provenance, dict) and res.provenance
    for name in ("outer_diameter", "eye_diameter", "inner_diameter", "hub_diameter", "blade_count",
                 "blade_angle_inlet", "blade_angle_outlet", "blade_width", "blade_width_inlet"):
        assert name in res.provenance, f"no provenance for {name}"
        v, src, conf = _prov(res, name)
        assert src in PROVENANCE_SOURCES, (name, src)
        assert 0.0 <= conf <= 1.0, (name, conf)
        # params are rounded (3 decimals) by the analyzer; provenance holds the raw value
        assert v == pytest.approx(getattr(res.params, name), rel=1e-3, abs=0.01), name
    assert res.rim is not None and hasattr(res.rim, "center") and hasattr(res.rim, "axes")
    assert isinstance(res.rectified, np.ndarray) and res.rectified.ndim == 2
    assert isinstance(res.debug, dict)
    assert isinstance(res.warnings, list)
    over = shrouded.draw_overlay(img, res)
    assert over.ndim == 3 and over.shape[:2] == img.shape[:2] and over.dtype == np.uint8


def _check_hub_contract(res, p):
    """Hub flange: 'prior' (0.3 D2, conf <= 0.1) or 'measured' within HUB_MEASURED_TOL."""
    v, src, conf = _prov(res, "hub_diameter")
    assert src in ("prior", "measured"), src
    if src == "prior":
        assert conf <= 0.1
        assert res.params.hub_diameter == pytest.approx(0.3 * res.params.outer_diameter, rel=1e-3)
    else:
        assert abs(res.params.hub_diameter - p.hub_diameter) / p.hub_diameter <= HUB_MEASURED_TOL, \
            f"hub: truth {p.hub_diameter} est {res.params.hub_diameter}"


RESULTS: list[dict] = []
CLUTTER_RESULTS: list[dict] = []


@pytest.fixture(scope="module", autouse=True)
def _summary(render_cache):
    t0 = time.perf_counter()
    yield
    print(f"\n[test_shrouded] module wall time {time.perf_counter() - t0:.0f} s")
    for label, rows_all in (("clean grid", RESULTS), ("clutter subset", CLUTTER_RESULTS)):
        if not rows_all:
            continue
        print(f"[test_shrouded] {label}: worst errors per preset (measured Z cases):")
        for name in PRESET_SPECS:
            rows = [r for r in rows_all if r["preset"] == name]
            if not rows:
                continue
            meas = [r for r in rows if r["z_src"] == "measured"]
            mx = lambda k, rs=meas: max((r[k] for r in rs), default=float("nan"))  # noqa: E731
            print(f"  {name:12s} n={len(rows)} Z measured {len(meas)} exact {sum(r['z_ok'] for r in meas)}"
                  f"  Zconf min {min((r['z_conf'] for r in meas), default=float('nan')):.2f}"
                  f"  D_eye(img) {mx('d_eye') * 100:.1f} %  D1 {mx('d1') * 100:.1f} %"
                  f"  hub(meas) {mx('hub', [r for r in meas if r['hub_src'] == 'measured']) * 100:.1f} %"
                  f"  beta1 {mx('beta1'):.1f}  beta2 {mx('beta2'):.1f} deg"
                  f"  b2conf {min((r['b2_conf'] for r in meas), default=float('nan')):.2f}"
                  f"..{mx('b2_conf'):.2f}  cam {mx('cam') * 100:.0f} %  rim px {mx('rim_px', rows):.1f}"
                  f"  t {mx('t', rows):.1f} s")


def _record(store, preset, Z, beta2, res, p, truth, dt):
    est = res.params
    cam = est.meta.get("camera_distance_mm")
    tr = truth["rim_ellipse"]
    rim_px = max(math.hypot(res.rim.center[0] - tr["center"][0], res.rim.center[1] - tr["center"][1]),
                 abs(res.rim.axes[0] - tr["axes"][0]))
    rec = {"preset": preset, "Z": Z, "beta2": beta2, "t": dt,
           "z_ok": est.blade_count == Z, "z_src": _prov(res, "blade_count")[1],
           "z_conf": _prov(res, "blade_count")[2],
           "d_eye": abs(_eye_measured_mm(res) - p.eye_diameter) / p.eye_diameter,
           "d1": abs(est.inner_diameter - p.inner_diameter) / p.inner_diameter,
           "hub": abs(est.hub_diameter - p.hub_diameter) / p.hub_diameter,
           "hub_src": _prov(res, "hub_diameter")[1],
           "beta1": abs(est.blade_angle_inlet - p.blade_angle_inlet),
           "beta2": abs(est.blade_angle_outlet - p.blade_angle_outlet),
           "b2_conf": _prov(res, "blade_angle_outlet")[2],
           "cam": (abs(cam - _cam_dist_truth(truth)) / _cam_dist_truth(truth)) if cam else float("nan"),
           "rim_px": rim_px}
    store.append(rec)
    return rec


# ---------------------------------------------------------------------------
# the clean grid: D2 + D_eye given
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Z,beta2,preset", CLEAN_GRID, ids=_ids(CLEAN_GRID))
def test_grid_clean(render_cache, Z, beta2, preset):
    img, truth, p = _grid_render(render_cache, Z, beta2, preset)
    assert truth["visible_blade_fraction"] > 0.3          # blade roots are visible through the eye
    t0 = time.perf_counter()
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    dt = time.perf_counter() - t0
    rec = _record(RESULTS, preset, Z, beta2, res, p, truth, dt)
    est = res.params
    _check_result_structure(res, img)
    assert dt < ANALYZE_TIME_BOUND_S, f"analyze_shrouded took {dt:.1f} s"

    # blade count
    v, src, conf = _prov(res, "blade_count")
    assert src == "measured" and conf >= Z_CONF_MIN, (src, conf)
    assert est.blade_count == Z, f"Z: truth {Z} est {est.blade_count}"

    # scale: both given -> 'user', and the perspective correction ran
    assert _prov(res, "outer_diameter")[1] == "user" and est.outer_diameter == pytest.approx(p.outer_diameter)
    assert _prov(res, "eye_diameter")[1] == "user" and est.eye_diameter == pytest.approx(p.eye_diameter)
    assert rec["d_eye"] <= D_EYE_TOL[preset], \
        f"image-measured D_eye: truth {p.eye_diameter} est {_eye_measured_mm(res):.1f}"
    cam = est.meta.get("camera_distance_mm")
    assert cam is not None and cam > 0
    if preset == "axial_close":
        assert rec["cam"] <= CAM_DIST_CLOSE_TOL, f"camera distance {cam:.0f} vs {_cam_dist_truth(truth):.0f} mm"

    _check_hub_contract(res, p)

    # blade geometry
    assert _prov(res, "inner_diameter")[1] in ("measured", "prior")
    assert rec["d1"] <= D1_TOL[preset], f"D1: truth {p.inner_diameter} est {est.inner_diameter}"
    assert _prov(res, "blade_angle_inlet")[1] == "measured"
    assert rec["beta1"] <= BETA1_TOL_DEG, f"beta1: truth {p.blade_angle_inlet} est {est.blade_angle_inlet}"
    v, src, conf = _prov(res, "blade_angle_outlet")
    assert src == "extrapolated"
    assert rec["beta2"] <= BETA2_EXTRAP_TOL_DEG[preset], f"beta2 (extrapolated): truth {beta2} est {v:.1f}"
    if preset == "axial_close":
        assert conf >= BETA2_CLOSE_CONF_MIN
    assert 0.0 < res.debug["visible_fraction"] <= 1.0
    assert est.meta.get("handedness") == "ccw"           # the CAD convention, as rendered

    # not observable axially -> priors with low confidence
    v, src, conf = _prov(res, "blade_width")
    assert src == "prior" and conf <= 0.1 and est.blade_width == pytest.approx(0.2 * est.outer_diameter, rel=1e-3)
    assert _prov(res, "blade_width_inlet")[1] == "prior"

    # the rim ellipse sits where the renderer put it (original image coordinates)
    if preset != "tilt10":          # tilted: the silhouette mixes the shroud rim and the backplate edge
        assert rec["rim_px"] <= RIM_PX_TOL, f"rim ellipse off by {rec['rim_px']:.1f} px"


# ---------------------------------------------------------------------------
# clutter subset: Z only when 'measured'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Z,beta2,preset", CLUTTER_GRID, ids=_ids(CLUTTER_GRID))
def test_grid_clutter(render_cache, Z, beta2, preset):
    img, truth, p = _grid_render(render_cache, Z, beta2, preset, clutter=True)
    assert truth["clutter"]
    t0 = time.perf_counter()
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    dt = time.perf_counter() - t0
    rec = _record(CLUTTER_RESULTS, preset, Z, beta2, res, p, truth, dt)
    est = res.params
    _check_result_structure(res, img)
    assert dt < ANALYZE_TIME_BOUND_S, f"analyze_shrouded took {dt:.1f} s"
    assert rec["d_eye"] <= D_EYE_TOL[preset], f"image-measured D_eye est {_eye_measured_mm(res):.1f}"
    _check_hub_contract(res, p)
    v, src, conf = _prov(res, "blade_count")
    assert src in ("measured", "prior"), src
    if src == "measured":
        assert est.blade_count == Z, f"Z: truth {Z} est {est.blade_count} ('measured', conf {conf:.2f})"
        assert _prov(res, "blade_angle_inlet")[1] == "measured"
        assert rec["beta1"] <= BETA1_CLUTTER_TOL_DEG, \
            f"beta1: truth {p.blade_angle_inlet} est {est.blade_angle_inlet}"
        assert _prov(res, "blade_angle_outlet")[1] == "extrapolated"
    else:
        # no blade pattern found: the prior must say so
        assert conf <= 0.1, conf
        assert est.blade_count == shrouded.PRIOR["Z"]
        assert _prov(res, "blade_angle_inlet")[1] == "prior" and _prov(res, "blade_angle_inlet")[2] <= 0.1
        assert _prov(res, "blade_angle_outlet")[1] == "prior" and _prov(res, "blade_angle_outlet")[2] <= 0.1
        assert any("no blade pattern" in w for w in res.warnings)


def test_clutter_subset_mostly_measured():
    """At least 2/3 of the clutter subset must have found the blade pattern ('measured')."""
    if not CLUTTER_RESULTS:
        pytest.skip("test_grid_clutter did not run in this session (deselected)")
    n_meas = sum(r["z_src"] == "measured" for r in CLUTTER_RESULTS)
    assert len(CLUTTER_RESULTS) == len(CLUTTER_GRID), "not every clutter case ran"
    assert n_meas >= CLUTTER_MEASURED_MIN_FRACTION * len(CLUTTER_RESULTS), \
        f"only {n_meas}/{len(CLUTTER_RESULTS)} clutter cases measured Z"
    assert all(r["z_ok"] for r in CLUTTER_RESULTS if r["z_src"] == "measured")


@pytest.mark.xfail(strict=True, reason="known miscount: Z9 b40 axial_far + clutter -> Z = 5 "
                                       "'measured' with confidence 0.69 (2026-11-02)")
def test_clutter_known_miscount(render_cache):
    Z, beta2, preset = CLUTTER_KNOWN_MISCOUNT
    img, truth, p = _grid_render(render_cache, Z, beta2, preset, clutter=True)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    v, src, conf = _prov(res, "blade_count")
    assert src != "measured" or res.params.blade_count == Z, (res.params.blade_count, src, conf)


# ---------------------------------------------------------------------------
# wrap-angle beta2 from marker side views
# ---------------------------------------------------------------------------
def _wrap_images(render_cache, Z, beta2):
    p = realistic_params(Z, beta2)
    img, truth = _render(render_cache, p, "axial_close", marker=True)
    assert truth["marker"]["visible_pixels"] > 0
    rim_images = []
    for az in SIDE_AZIMUTHS:
        rim_img, rim_truth = _render(render_cache, p, "side", marker=True, azimuth_deg=az, seed=int(az))
        assert rim_truth["marker"]["visible_pixels"] > 0
        rim_images.append(rim_img)
    return img, rim_images, p


def _analyze_wrap(render_cache, Z, beta2):
    img, rim_images, p = _wrap_images(render_cache, Z, beta2)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter,
                                    rim_images=rim_images, marker_hint="bright")
    return res, p, img


@pytest.mark.parametrize("Z,beta2", WRAP_CASES, ids=[f"Z{z}-b2_{b:g}" for z, b in WRAP_CASES])
def test_wrap_angle_beta2(render_cache, Z, beta2):
    res, p, img = _analyze_wrap(render_cache, Z, beta2)
    est = res.params
    _check_result_structure(res, img)
    assert est.blade_count == Z and _prov(res, "blade_count")[1] == "measured"
    v, src, conf = _prov(res, "blade_angle_outlet")
    assert src == "wrap", f"beta2 provenance {src}, expected 'wrap' (warnings: {res.warnings})"
    assert conf >= BETA2_WRAP_CONF_MIN, conf
    assert abs(est.blade_angle_outlet - beta2) <= BETA2_WRAP_TOL_DEG, \
        f"beta2 (wrap): truth {beta2} est {est.blade_angle_outlet}"
    wrap = res.debug["wrap"]
    wrap_true = geometry.wrap_from_betas(p.r1, p.r2, p.blade_angle_inlet, p.blade_angle_outlet)
    assert abs(wrap["wrap_deg"] - wrap_true) <= 5.0, (wrap["wrap_deg"], wrap_true)
    assert len(res.debug["rim_images"]) == len(SIDE_AZIMUTHS)
    # blade width from the rim gap when the user does not give it
    v, src, conf = _prov(res, "blade_width")
    assert src == "measured", src
    assert abs(est.blade_width - p.b2) / p.b2 <= B2_RIM_GAP_TOL, f"b2: truth {p.b2} est {est.blade_width}"


@pytest.mark.xfail(strict=True, reason="known bias: Z13 b30 wrap 99.9 deg vs truth 90.9 -> beta2 "
                                       "26.55 deg, 3.45 deg off (2026-11-02)")
def test_wrap_angle_known_bias(render_cache):
    Z, beta2 = WRAP_KNOWN_BIAS
    res, p, img = _analyze_wrap(render_cache, Z, beta2)
    assert _prov(res, "blade_angle_outlet")[1] == "wrap"
    assert abs(res.params.blade_angle_outlet - beta2) <= BETA2_WRAP_TOL_DEG, res.params.blade_angle_outlet


# ---------------------------------------------------------------------------
# factory params: no blade visible from far away -> Z is a prior, by construction
# ---------------------------------------------------------------------------
def test_factory_axial_far_blade_count_is_prior(render_cache):
    p = SHROUDED_PARAMS_FACTORY()
    assert p.r1 > p.r_eye                                 # LE hidden under the shroud lip
    img, truth = _render(render_cache, p, "axial_far")
    assert truth["visible_blade_fraction"] == 0.0
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    _check_result_structure(res, img)
    v, src, conf = _prov(res, "blade_count")
    assert src == "prior" and conf <= 0.1 and res.params.blade_count == shrouded.PRIOR["Z"]
    for name in ("blade_angle_inlet", "blade_angle_outlet"):
        assert _prov(res, name)[1] == "prior" and _prov(res, name)[2] <= 0.1
    v, src, conf = _prov(res, "inner_diameter")
    assert src == "prior" and res.params.inner_diameter == pytest.approx(res.params.eye_diameter, rel=1e-3)
    assert any("no blade pattern" in w for w in res.warnings)
    # a user hint is taken as 'user' then (with the hint's value, not the prior)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter,
                                    blade_count_hint=13)
    v, src, conf = _prov(res, "blade_count")
    assert src == "user" and conf >= 0.8 and res.params.blade_count == 13


# ---------------------------------------------------------------------------
# scale fallbacks
# ---------------------------------------------------------------------------
def test_no_scale_prior_keeps_ratios(render_cache):
    """No D2, no D_eye, no reference disc: D2 := 300 'prior' (conf 0.05) and the eye/rim ratio
    is still measured (within 0.9 % at axial_far)."""
    img, truth, p = _grid_render(render_cache, 11, 40.0, "axial_far")
    res = shrouded.analyze_shrouded(img)
    est = res.params
    v, src, conf = _prov(res, "outer_diameter")
    assert src == "prior" and conf <= 0.1 and est.outer_diameter == pytest.approx(shrouded.PRIOR["D2_mm"])
    assert _prov(res, "eye_diameter")[1] == "prior"       # relative to a prior D2
    assert abs(est.eye_diameter / est.outer_diameter - p.eye_diameter / p.outer_diameter) <= 0.02 * 0.6
    assert est.blade_count == 11 and _prov(res, "blade_count")[1] == "measured"
    assert abs(est.blade_angle_inlet - p.blade_angle_inlet) <= BETA1_TOL_DEG
    v, src, conf = _prov(res, "blade_width")
    assert src == "prior" and conf <= 0.1 and est.blade_width == pytest.approx(0.2 * est.outer_diameter, rel=1e-3)
    assert est.meta.get("camera_distance_mm") is None
    assert any("no scale" in w for w in res.warnings)


def test_eye_diameter_only_far_camera(render_cache):
    """Only D_eye known, far camera: D2 'measured' within 2 % (0.9 % achieved), with the
    'far camera assumed' warning."""
    img, truth, p = _grid_render(render_cache, 11, 40.0, "axial_far")
    res = shrouded.analyze_shrouded(img, eye_diameter_mm=p.eye_diameter)
    est = res.params
    assert _prov(res, "eye_diameter")[1] == "user" and est.eye_diameter == pytest.approx(p.eye_diameter)
    v, src, conf = _prov(res, "outer_diameter")
    assert src == "measured" and conf <= 0.6
    assert abs(est.outer_diameter - p.outer_diameter) / p.outer_diameter <= 0.02
    assert est.blade_count == 11
    assert any("far camera" in w for w in res.warnings)


def test_d2_only_close_camera_bias(render_cache):
    """Documented limitation: with D2 only at axial_close the eye ring (~35 mm nearer the camera
    than the rim) is reported 6..13 % too large (+13.1 % here) because no perspective
    correction is possible from a single length; the analyzer warns about it.  Giving D_eye
    too (test_grid_clean) removes the bias."""
    img, truth, p = _grid_render(render_cache, 11, 40.0, "axial_close")
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter)
    est = res.params
    assert _prov(res, "eye_diameter")[1] == "measured"
    bias = (est.eye_diameter - p.eye_diameter) / p.eye_diameter
    assert 0.0 < bias <= 0.16, f"D_eye bias {bias * 100:+.1f} % (expected the far-camera over-estimate)"
    assert est.meta.get("camera_distance_mm") is None
    assert any("far camera" in w for w in res.warnings)
    assert est.blade_count == 11
    assert abs(est.blade_angle_inlet - p.blade_angle_inlet) <= BETA1_TOL_DEG


# ---------------------------------------------------------------------------
# real photo
# ---------------------------------------------------------------------------
def test_real_axial_sample():
    """samples/real_axial_eye.jpg (phone photo through the eye, tilted, cluttered desk; 8 blades
    by eye): rim + inlet ring found, Z = 8 'measured', beta1 plausible, blades sweep clockwise,
    runtime < 15 s, overlay of the same shape.  No ground truth beyond the blade count."""
    img = cv2.imread(str(REAL_AXIAL), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    t0 = time.perf_counter()
    res = shrouded.analyze_shrouded(img)
    dt = time.perf_counter() - t0
    assert dt < REAL_TIME_BOUND_S, f"analyze_shrouded took {dt:.1f} s"
    _check_result_structure(res, img)
    assert res.eye is not None
    h, w = img.shape
    assert 0.3 * w < res.rim.center[0] < 0.7 * w and 0.3 * h < res.rim.center[1] < 0.7 * h
    assert res.rim.axes[0] > 0.3 * w and res.eye.axes[0] < res.rim.axes[0]
    v, src, conf = _prov(res, "blade_count")
    assert src == "measured" and res.params.blade_count == 8, (res.params.blade_count, src, conf)
    assert res.debug["expected_from_eye"] == 8
    v, src, conf = _prov(res, "blade_angle_inlet")
    assert src == "measured" and 20.0 <= res.params.blade_angle_inlet <= 40.0, (v, conf)
    print(f"\n[test_shrouded] real photo: Z {res.params.blade_count} (conf {_prov(res, 'blade_count')[2]:.2f}), "
          f"beta1 {res.params.blade_angle_inlet:.1f} deg (conf {conf:.2f}), beta2 "
          f"{res.params.blade_angle_outlet:.1f} ({_prov(res, 'blade_angle_outlet')[1]}, conf "
          f"{_prov(res, 'blade_angle_outlet')[2]:.2f}), {dt:.1f} s")
    v, src, conf = _prov(res, "blade_angle_outlet")
    assert src == "extrapolated" and conf <= 0.3          # ~1/3 of the span visible, 1 blade traced
    assert res.params.meta.get("handedness") == "cw"
    v, src, conf = _prov(res, "outer_diameter")
    assert src == "prior" and conf <= 0.1 and v == pytest.approx(300.0)
    over = shrouded.draw_overlay(img, res)
    assert over.shape[:2] == img.shape[:2] and over.ndim == 3
    # with D2 given everything scales consistently and Z does not change
    res2 = shrouded.analyze_shrouded(img, outer_diameter_mm=320.0)
    assert _prov(res2, "outer_diameter")[1] == "user"
    assert res2.params.eye_diameter / res2.params.outer_diameter == pytest.approx(
        res.params.eye_diameter / res.params.outer_diameter, rel=0.02)
    assert res2.params.blade_count == 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_analyze_shrouded(tmp_path, render_cache, capsys):
    from fanscan import cli, render3d

    img, truth, p = _grid_render(render_cache, 11, 40.0, "axial_close")
    png = tmp_path / "shot.png"
    render3d.save(png, img)
    out = tmp_path / "params.json"
    overlay = tmp_path / "overlay.png"
    rc = cli.main(["analyze", str(png), "--mode", "shrouded", "--outer-diameter-mm", "300",
                   "--eye-diameter-mm", "180", "--out", str(out), "--overlay", str(overlay)])
    assert rc == 0
    est = ImpellerParams.load(out)
    assert est.blade_count == 11 and est.outer_diameter == 300.0 and est.eye_diameter == 180.0
    assert abs(est.blade_angle_outlet - 40.0) <= BETA2_EXTRAP_TOL_DEG["axial_close"]
    assert overlay.exists() and cv2.imread(str(overlay)) is not None
    text = capsys.readouterr().out
    assert "blade_angle_outlet" in text and "extrapolated" in text and "user" in text
