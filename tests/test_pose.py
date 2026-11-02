"""Tests for fanscan.pose (rim / eye ellipse detection, rectification, perspective) and the
``--mode auto`` rule of the CLI.

Achieved numbers (fanscan.pose, this machine):
  synthetic ellipses on procedural clutter (5 cases incl. aspect 0.6, angle 160 deg, a cable
  crossing the rim, blur 1 px, noise sigma 6): rim centre <= 0.2 px, semi-axes <= 0.72 px,
  angle <= 0.26 deg, score >= 0.99; eye centre <= 0.4 px, semi-axes <= 1.2 px (either edge of
  the 4 px ring band), score >= 0.94.  0.6-1.2 s per image.
  samples/real_axial_eye.jpg (1440x1425): rim a=659.4 b=650.4 c=(708.9, 711.9) score 0.99,
  eye a=441.2 b=432.4 c=(675.0, 672.2) score 0.99, ~1.0 s; tilt from rim 9.4 deg.
  samples/real_side_oblique.jpg: best ellipse = inlet-ring top, aspect 0.11, score 0.95; the
  very flat outer rim (b/a ~ 0.13, far side hidden behind the cone) is NOT recovered.
  render3d axial_close / tilt(10) on clutter: see the tolerances in the tests below.
Timing of the whole file: ~12 s plus the shared mesh build (~15 s) if test_render3d did not
run first in the same process.
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from fanscan import pose
from fanscan.params import SHROUDED_PARAMS_FACTORY

ROOT = Path(__file__).resolve().parents[1]
REAL_AXIAL = ROOT / "samples" / "real_axial_eye.jpg"
REAL_SIDE = ROOT / "samples" / "real_side_oblique.jpg"


# ---------------------------------------------------------------------------
# synthetic images: clutter + a shrouded-impeller-like family of ellipses
# ---------------------------------------------------------------------------
def _clutter(size, seed):
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w), 165, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    img += 12 * np.sin(xx / 37.0 + 3 * np.sin(yy / 90.0)) + 6 * np.sin(yy / 23.0)
    for _ in range(6):
        x0, y0 = rng.integers(0, w - 50), rng.integers(0, h - 50)
        x1, y1 = x0 + rng.integers(40, w // 2), y0 + rng.integers(40, h // 2)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), float(rng.integers(20, 70)), -1)
    for _ in range(8):
        pts = np.cumsum(rng.normal(0, 25, (12, 2)), axis=0) + rng.uniform(0, [w, h])
        cv2.polylines(img, [pts.astype(np.int32)], False, 25.0, int(rng.integers(2, 5)), cv2.LINE_AA)
    cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(20, 60)), 250.0, -1)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _synthetic(size, center, a, b, angle, ratio_eye, seed, noise=6.0, blur=1.0):
    """Bright shroud ellipse (a, b, angle) with a dark 4 px inlet-ring band at ratio_eye,
    11 dark blade wedges, hub flange + bore, and a cable crossing the rim, on clutter."""
    f = _clutter(size, seed).astype(np.float32)
    cv2.ellipse(f, (center, (2 * a, 2 * b), angle), 215.0, -1, cv2.LINE_AA)
    ae, be = a * ratio_eye, b * ratio_eye
    cv2.ellipse(f, (center, (2 * ae, 2 * be), angle), 90.0, -1, cv2.LINE_AA)
    cv2.ellipse(f, (center, (2 * ae - 8, 2 * be - 8), angle), 150.0, -1, cv2.LINE_AA)
    for i in range(11):
        t0 = 2 * math.pi * i / 11
        p = [(center[0] + 0.35 * ae * math.cos(t0 + k * 0.4), center[1] + 0.35 * be * math.sin(t0 + k * 0.4))
             for k in range(3)]
        p += [(center[0] + 0.95 * ae * math.cos(t0 + 1.2 + k * 0.1),
               center[1] + 0.95 * be * math.sin(t0 + 1.2 + k * 0.1)) for k in range(2)]
        cv2.fillPoly(f, [np.array(p, np.int32)], 120.0, cv2.LINE_AA)
    cv2.ellipse(f, (center, (0.6 * ae, 0.6 * be), angle), 175.0, -1, cv2.LINE_AA)
    cv2.ellipse(f, (center, (0.35 * ae, 0.35 * be), angle), 30.0, -1, cv2.LINE_AA)
    cv2.line(f, (int(center[0] - a - 60), int(center[1] + 0.3 * b)),
             (int(center[0] + 0.2 * a), int(center[1] + b + 80)), 25.0, 4, cv2.LINE_AA)
    if blur:
        f = cv2.GaussianBlur(f, (0, 0), blur)
    f += np.random.default_rng(seed + 100).normal(0, noise, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


CASES = [  # size, centre, a, b, angle, eye ratio, seed
    ((1200, 1200), (610, 590), 480, 480, 0, 0.62, 1),
    ((1200, 1200), (600, 620), 470, 420, 20, 0.6, 2),
    ((1000, 800), (520, 390), 330, 240, 70, 0.55, 4),
    ((1200, 1200), (560, 640), 500, 300, 160, 0.65, 5),
    ((900, 1200), (450, 600), 300, 290, 10, 0.75, 6),
]
IDS = [f"a{a}-b{b}-ang{ang}" for _, _, a, b, ang, _, _ in CASES]


def _ang_err(e, truth, aspect):
    return 0.0 if aspect > 0.97 else abs((e.angle_deg - truth + 90) % 180 - 90)


@pytest.mark.parametrize("size,c,a,b,ang,ratio,seed", CASES, ids=IDS)
def test_synthetic_on_clutter(size, c, a, b, ang, ratio, seed):
    img = _synthetic(size, c, a, b, ang, ratio, seed)
    t0 = time.perf_counter()
    ells = pose.find_ellipses(img)
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"find_ellipses took {dt:.1f} s"
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    assert rim is not None and eye is not None
    # SPEC2: found within 1 px / 1 deg
    assert math.hypot(rim.center[0] - c[0], rim.center[1] - c[1]) < 1.0
    assert abs(rim.a - a) < 1.0 and abs(rim.b - b) < 1.0
    assert _ang_err(rim, ang, b / a) < 1.0
    assert rim.score > 0.9 and rim.n_inliers > 0
    # the eye ring is a 4 px dark band: either edge is a valid detection
    assert min(abs(eye.a - (a * ratio - d)) for d in (0, 4)) < 1.5
    assert min(abs(eye.b - (b * ratio - d)) for d in (0, 4)) < 1.5
    assert math.hypot(eye.center[0] - c[0], eye.center[1] - c[1]) < 1.0
    assert eye.score > 0.8
    # acos(b/a) is ill-conditioned near a circle (0.2 px of axis error -> 1.5 deg): only
    # check the tilt angle for clearly elliptical rims, and a loose bound for circles
    tilt_truth = math.degrees(math.acos(b / a))
    if b / a < 0.97:
        assert abs(pose.tilt_from_ellipse(rim) - tilt_truth) < 1.0
    else:
        assert pose.tilt_from_ellipse(rim) < 3.0


def test_no_ellipse_in_noise():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (500, 600), dtype=np.uint8)
    ells = pose.find_ellipses(img)
    assert all(e.score < 0.5 for e in ells)
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    assert eye is None
    assert pose.pick_impeller_ellipses([], img.shape) == (None, None)
    assert pose.find_ellipses(np.full((300, 300), 128, np.uint8)) == []


def test_accepts_bgr_and_small_images():
    img = _synthetic((600, 600), (300, 300), 240, 220, 30, 0.6, 9)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rim, _ = pose.pick_impeller_ellipses(pose.find_ellipses(bgr), bgr.shape)
    assert rim is not None and abs(rim.a - 240) < 1.5
    assert pose.find_ellipses(np.zeros((20, 20), np.uint8)) == []


# ---------------------------------------------------------------------------
# real samples
# ---------------------------------------------------------------------------
def test_real_axial_sample():
    img = cv2.imread(str(REAL_AXIAL))
    assert img is not None
    t0 = time.perf_counter()
    ells = pose.find_ellipses(img)
    dt = time.perf_counter() - t0
    assert dt < 4.0, f"find_ellipses took {dt:.1f} s on the 1440x1425 sample"
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    assert rim is not None and eye is not None
    assert 640 < rim.a < 680 and 630 < rim.b < 670            # outer shroud rim
    assert 690 < rim.center[0] < 730 and 695 < rim.center[1] < 730
    assert 425 < eye.a < 455 and 420 < eye.b < 445            # inlet ring
    assert rim.score > 0.9 and eye.score > 0.9
    assert 0.6 < eye.a / rim.a < 0.72
    assert 5 < pose.tilt_from_ellipse(rim) < 15
    # rectification makes the rim a centred circle; the eye stays roughly concentric
    rect, M, ppu = pose.rectify(img, rim)
    assert ppu == pytest.approx(rim.a)
    rr = pose.transform_ellipse(rim, M)
    ee = pose.transform_ellipse(eye, M)
    assert abs(rr.aspect - 1.0) < 1e-3 and abs(rr.a - rim.a) < 1e-3
    assert abs(rr.center[0] - rect.shape[1] / 2) < 1e-6 and abs(rr.center[1] - rect.shape[0] / 2) < 1e-6
    assert math.hypot(ee.center[0] - rr.center[0], ee.center[1] - rr.center[1]) < 0.15 * rim.a
    over = pose.draw_ellipses(img, ells)
    assert over.shape == img.shape and over.dtype == np.uint8


def test_real_side_sample_reports_inlet_ring():
    img = cv2.imread(str(REAL_SIDE))
    assert img is not None
    ells = pose.find_ellipses(img)
    assert ells, "at least the inlet-ring top ellipse must be found"
    top = max(ells, key=lambda e: e.score)
    assert top.aspect < 0.2 and top.score > 0.8                 # very flat: elevation ~6 deg
    assert pose.tilt_from_ellipse(top) > 75
    # a side view must not be mistaken for an axial one: the 'rim' is flatter than 0.3
    rim, _ = pose.pick_impeller_ellipses(ells, img.shape)
    assert rim is not None and rim.aspect < 0.3


# ---------------------------------------------------------------------------
# rectification / perspective (closed form)
# ---------------------------------------------------------------------------
def test_rectify_and_affine_utilities():
    img = np.zeros((900, 1100), np.uint8)
    rim = pose.EllipseFit((520.0, 460.0), (400.0, 260.0), 25.0)
    eye = pose.EllipseFit((530.0, 455.0), (260.0, 169.0), 25.0)
    rect, M, ppu = pose.rectify(img, rim)
    assert M.shape == (2, 3) and rect.shape[0] == rect.shape[1] and ppu == 400.0
    rr, ee = pose.transform_ellipse(rim, M), pose.transform_ellipse(eye, M)
    assert abs(rr.aspect - 1) < 1e-3 and abs(rr.a - 400) < 1e-3
    assert abs(rr.center[0] - rect.shape[1] / 2) < 1e-6
    assert abs(ee.aspect - 1) < 1e-3 and abs(ee.a - 260) < 1e-3
    assert abs(pose.tilt_from_ellipse(rim) - math.degrees(math.acos(260 / 400))) < 1e-9
    # affine round trip and custom output size
    pts = np.array([[10.0, 20.0], [500.0, 700.0], [-3.0, 1000.0]])
    back = pose.apply_affine(pose.invert_affine(M), pose.apply_affine(M, pts))
    assert np.allclose(back, pts, atol=1e-9)
    rect2, M2, _ = pose.rectify(img, rim, out_size=(640, 480))
    assert rect2.shape == (480, 640)
    assert np.allclose(pose.apply_affine(M2, np.array([rim.center])), [[320.0, 240.0]], atol=1e-9)
    # EllipseFit conversions
    e = pose.EllipseFit.from_cv(((10.0, 20.0), (30.0, 80.0), 15.0))
    assert e.a == 40.0 and e.b == 15.0 and e.angle_deg == pytest.approx(105.0)
    assert e.to_cv()[1] == (80.0, 30.0)
    d = e.to_dict()
    assert d["axes"] == [40.0, 15.0] and d["center"] == [10.0, 20.0]


def test_estimate_perspective_scale():
    f, R, k, D, dz = 1150.0, 150.0, 0.6, 450.0, 38.0
    r = pose.EllipseFit((0, 0), (f * R / D, f * R / D), 0)
    e = pose.EllipseFit((0, 0), (f * k * R / (D - dz),) * 2, 0)
    assert abs(pose.estimate_perspective_scale(r, e, 0.0, dz, k) - D) < 1e-6
    assert abs(pose.estimate_perspective_scale(r, e, 0.0, dz, rim_radius=R, eye_radius=k * R) - D) < 1e-6
    assert pose.estimate_perspective_scale(r, e, 0.0, dz) is None            # ratio unknown
    assert pose.estimate_perspective_scale(r, e, dz, 0.0, k) is None         # eye below the rim
    far = pose.EllipseFit((0, 0), (f * k * R / D * 1.01,) * 2, 0)
    assert pose.estimate_perspective_scale(r, far, 0.0, dz, k) is None       # far camera: ill-conditioned
    assert pose.scale_at_height(D, 0.0, dz) == pytest.approx(D / (D - dz))
    assert pose.scale_at_height(D, 0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# render3d images on clutter (shared mesh with test_render3d via render3d._MESH_CACHE)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def factory_mesh():
    from fanscan import render3d
    return render3d.mesh_from_params(SHROUDED_PARAMS_FACTORY())


def test_render3d_axial_close_on_clutter(factory_mesh):
    """Rim within 2 px of the truth rim ellipse; the eye (inlet ring, either edge) found."""
    from fanscan import render3d

    p = SHROUDED_PARAMS_FACTORY()
    img, truth = render3d.render_params(p, "axial_close", size=(1200, 1200), mesh=factory_mesh,
                                        clutter=True, noise_sigma=4.0, seed=2)
    ells = pose.find_ellipses(img)
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    assert rim is not None and eye is not None
    tr = truth["rim_ellipse"]
    assert math.hypot(rim.center[0] - tr["center"][0], rim.center[1] - tr["center"][1]) <= 2.0
    assert abs(rim.a - tr["axes"][0]) <= 2.0 and abs(rim.b - tr["axes"][1]) <= 2.0
    te = truth["eye_ellipse"]
    # inlet ring: inner edge (r_eye) or outer edge (r_eye + shroud_thickness) -> within 4 %
    assert abs(eye.a - te["axes"][0]) / te["axes"][0] < 0.04
    assert math.hypot(eye.center[0] - te["center"][0], eye.center[1] - te["center"][1]) <= 4.0


def test_render3d_tilt_on_clutter(factory_mesh):
    """Tilt 10 deg: the rim ellipse is found and its foreshortening gives the tilt within 3 deg.
    (In a tilted view the near side of the silhouette is the backplate / rim underside, so the
    fitted rim may blend the two edges: centre/axes tolerance 4 px.)"""
    from fanscan import render3d

    p = SHROUDED_PARAMS_FACTORY()
    img, truth = render3d.render_params(p, "tilt", size=(1200, 1200), mesh=factory_mesh,
                                        clutter=True, noise_sigma=4.0, seed=3, tilt_deg=10.0)
    ells = pose.find_ellipses(img)
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    assert rim is not None
    tr = truth["rim_shroud_ellipse"]
    assert math.hypot(rim.center[0] - tr["center"][0], rim.center[1] - tr["center"][1]) <= 4.0
    assert abs(rim.a - tr["axes"][0]) <= 4.0 and abs(rim.b - tr["axes"][1]) <= 4.0
    assert abs(pose.tilt_from_ellipse(rim) - 10.0) < 3.0
    assert eye is not None
    assert abs(eye.a - truth["eye_ellipse"]["axes"][0]) / truth["eye_ellipse"]["axes"][0] < 0.05


# ---------------------------------------------------------------------------
# CLI --mode auto rule (pose-based)
# ---------------------------------------------------------------------------
def test_cli_detect_mode():
    from fanscan import cli, synth
    from fanscan.params import DEFAULT_PARAMS

    img, _ = synth.render(DEFAULT_PARAMS, px_per_mm=4.0, ref_diameter_mm=30.0, noise_sigma=8.0)
    assert cli.detect_mode(img, 30.0) == "open"          # white reference disc present
    assert cli.detect_mode(img, None) == "open"          # open impeller: no eye ellipse
    real = cv2.imread(str(REAL_AXIAL), cv2.IMREAD_GRAYSCALE)
    assert cli.detect_mode(real, None) == "shrouded"     # rim + inlet ring, no disc
