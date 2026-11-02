"""End-to-end tests of the shrouded-mode analyzer (fanscan.shrouded) on render3d images, plus
the ``analyze --mode shrouded`` / ``pipeline3d`` CLI paths.

The whole file is skipped (not failed) only when ``fanscan.shrouded`` cannot be imported.

TOLERANCE TABLE  -- TODO-verify: the asserts use the SPEC2 targets; the 'achieved' column is a
SNAPSHOT measured on 2026-11-02 against fanscan/shrouded.py while it was still being written
(two runs 20 min apart gave different tilt20 numbers).  The integrator re-runs this file
against the final analyzer and replaces the column (loosen an assert with a comment where
needed; never skip a case silently).  Worst case over the 18 cases of each preset:

    quantity                         preset(s)                    SPEC2 target   achieved (snapshot)
    ------------------------------------------------------------------------------------------------
    blade count Z                    all four                     exact          far 17/18, close 18/18,
                                                                                 tilt10 18/18, tilt20 13/18
    D_eye (D2 given)                 all four                     within 4 %     far 2.5 %, close 13.1 % (see
                                                                                 gaps), tilt10 5.8 %, tilt20 3.7 %
    hub (D2 given)                   all four                     within 4 %     10.0 % in ALL presets = the
                                                                                 0.3 D2 prior: the 16 mm flange
                                                                                 is never detected in renders
    beta1 (measured)                 all four                     within 5 deg   far 1.1, close 1.1, tilt10 1.8,
                                                                                 tilt20 43 (19 in the 1st run)
    beta2 (extrapolated)             axial_close only             within 8 deg   9.2 deg (far 10.0, tilt10 5.7)
    beta2 (wrap, 3 side images)      axial_close + side x3        within 3 deg   not reached: marker not found
                                                                                 in the axial image (see gaps)
    D_eye/D2, hub/D2 (no scale)      axial_far                    within 4 %     D_eye ok; hub = prior (10 %)
    D2 from D_eye only               axial_far                    within 4 %     ok
    real sample Z (no truth)         samples/real_axial_eye.jpg   10..12         11, D2 prior 300, 2.1 s
    runtime per analyze_shrouded     1000 px grid image           < 5 s          1.8-4.7 s (tilt20 slowest)
    whole file                                                    ~4 min suite   404 s incl. 112 s mesh build
                                                                                 (217 s grid with cached meshes)
    pass/fail at the snapshot: 6 passed, 77 failed (70 grid: 32 hub, 32 D_eye, 6 Z)

Grid (SPEC2): presets {axial_far, axial_close, tilt(10), tilt(20)} x Z {9, 11, 13}
x beta2 {30, 40, 55} x clutter {off, on} = 72 cases.

KNOWN INTEGRATION GAPS (verified against fanscan/shrouded.py as it landed on 2026-11-02):
* wrap-angle tests + the pipeline3d wrap path: shrouded._find_marker_axial searches the
  rectified annulus (1.04 r_eye, 0.985 a_rim), but render3d.marker_box is a cube straddling the
  rim TOP edge (r in [r2 - s, r2]) rising s above it, so under the close camera it projects at
  ~1.0 a_rim and is never found ("marker not found in the axial image; wrap-angle beta2
  skipped") at 1000 or 1200 px and for 7.5 or 12 mm markers.  Fix on either side: place the
  marker flat on the shroud face inside the rim (render3d) or widen the window to ~1.03 a_rim
  (shrouded).  On the 'side' image at camera azimuth 270 the gap-band detector also locked onto
  the inlet-ring edge (gap 429..704 instead of 496..704), so that image's marker was missed.
* D_eye at axial_close with D2 only: the eye ring is ~35 mm nearer the camera than the rim; with
  a single known length SPEC2 says "assume a far camera", which leaves a +6..13 % bias (measured
  +5.9 % and +13.1 % clusters).  Either give eye_diameter_mm too (then it is 'user') or restrict
  the 4 % D_eye check to axial_far / tilt presets.

GRID BASE PARAMS: SHROUDED_PARAMS_FACTORY() with inner_diameter = 130 (LE at r1 = 65 < r_eye
= 90, as in the real sample where the leading edges sit well inside the eye) and a short hub
flange (hub_height = 16).  With the factory's r1 = 95 > r_eye = 90 no blade at all is visible
through the eye from 'axial_far' (truth visible_blade_fraction = 0.00), so Z could not be
found by construction; the factory params are exercised separately in
test_factory_params_axial_close.  Visible blade fraction of the base params (truth):
axial_far 0.34, axial_close 0.57, tilt 0.74.

Runtime: 9 distinct (Z, beta2) CAD meshes (~14 s each; a process pool is used but gave no
speed-up on the 4-vCPU dev box) are built on the FIRST run and cached as .npz under pytest's
cache directory (.pytest_cache/d/fanscan_meshes, keyed by params + fanscan/cad.py source hash,
so a cad change rebuilds them); later runs load them in < 1 s.  Then 72 renders at GRID_SIZE
px (~1 s each) + analysis.  Set FANSCAN_GRID=quick to run the 8-case diagonal subset while
iterating (the default is the full SPEC2 grid; the environment variable is an explicit opt-in,
not a silent skip).
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

shrouded = pytest.importorskip("fanscan.shrouded")

from fanscan import render3d  # noqa: E402
from fanscan.params import SHROUDED_PARAMS_FACTORY, ImpellerParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL_AXIAL = ROOT / "samples" / "real_axial_eye.jpg"

# ---- tolerances (SPEC2 targets; see the header table) ------------------------------------
D_TOL = 0.04                 # D2 / D_eye / hub relative error when D2 is given
BETA1_TOL_DEG = 5.0
BETA2_EXTRAP_TOL_DEG = 8.0   # axial_close only
BETA2_WRAP_TOL_DEG = 3.0
RATIO_TOL = 0.04             # D_eye/D2, hub/D2 when no scale is given
ANALYZE_TIME_BOUND_S = 5.0

GRID_SIZE = 1000             # render size [px] for the grid (speed; SPEC2 renders are 1200)
GRID_INNER_DIAMETER = 130.0
GRID_HUB_DIAMETER = 100.0    # != 0.3 * D2 = 90 (the analyzer's hub prior), so the hub check is real
GRID_HUB_HEIGHT = 16.0
Z_VALUES = (9, 11, 13)
BETA2_VALUES = (30.0, 40.0, 55.0)
PRESET_SPECS = {             # name -> (render3d preset, extra render kwargs)
    "axial_far": ("axial_far", {}),
    "axial_close": ("axial_close", {}),
    "tilt10": ("tilt", {"tilt_deg": 10.0}),
    "tilt20": ("tilt", {"tilt_deg": 20.0}),
}
CLUTTER_VALUES = (False, True)

_FULL = list(itertools.product(Z_VALUES, BETA2_VALUES, PRESET_SPECS, CLUTTER_VALUES))
_QUICK = [(9, 30.0, "axial_far", False), (11, 40.0, "axial_close", True), (13, 55.0, "tilt10", False),
          (9, 55.0, "tilt20", True), (13, 30.0, "axial_close", False), (11, 30.0, "axial_far", True),
          (11, 55.0, "tilt20", False), (13, 40.0, "tilt10", True)]
GRID = _QUICK if os.environ.get("FANSCAN_GRID", "").lower() == "quick" else _FULL
GRID_IDS = [f"Z{z}-b2_{b:g}-{p}-{'clutter' if c else 'plain'}" for z, b, p, c in GRID]
PROVENANCE_SOURCES = {"measured", "extrapolated", "wrap", "prior", "user"}


# ---------------------------------------------------------------------------
# params / mesh helpers
# ---------------------------------------------------------------------------
def grid_params(Z: int, beta2: float) -> ImpellerParams:
    p = dataclasses.replace(SHROUDED_PARAMS_FACTORY(), blade_count=Z, blade_angle_outlet=beta2,
                            inner_diameter=GRID_INNER_DIAMETER, hub_diameter=GRID_HUB_DIAMETER,
                            hub_height=GRID_HUB_HEIGHT, meta={})
    p.validate()
    return p


MESH_TOL = 0.5


def _params_key(d: dict) -> str:
    return json.dumps({k: v for k, v in d.items() if k != "meta"}, sort_keys=True, default=str)


def _mesh_key(d: dict) -> str:
    """Cache key: params (minus meta) + tessellation tolerance + fanscan/cad.py source."""
    from fanscan import cad

    src = Path(cad.__file__).read_bytes()
    return hashlib.sha1(src + _params_key(d).encode() + f"|{MESH_TOL}".encode()).hexdigest()[:20]


def _build_mesh(d: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Worker: params dict -> (dict, V, F).  Runs in a separate process (cadquery)."""
    from fanscan import cad
    from fanscan.params import ImpellerParams

    V, F = cad.mesh_from_params(ImpellerParams.from_dict(d), MESH_TOL)
    return d, np.asarray(V, np.float64), np.asarray(F, np.int64)


def _seed_render3d_cache(d: dict, V: np.ndarray, F: np.ndarray) -> None:
    """Let render3d.render_params(mesh=None) (e.g. the CLI) reuse a mesh we already hold."""
    render3d._MESH_CACHE[_params_key(d) + f"|{MESH_TOL}"] = (V, F)


@pytest.fixture(scope="session")
def mesh_cache_dir(request) -> Path | None:
    try:
        return Path(request.config.cache.mkdir("fanscan_meshes"))
    except Exception:  # pragma: no cover - e.g. ``-p no:cacheprovider``
        return None


def load_or_build_meshes(todo: dict, cache_dir: Path | None) -> dict:
    """{key: (V, F)} for ``{key: params_dict}``: from the .npz cache, else built (in a process
    pool when several are missing) and stored."""
    out: dict = {}
    missing: dict = {}
    for key, d in todo.items():
        f = None if cache_dir is None else cache_dir / f"{_mesh_key(d)}.npz"
        if f is not None and f.exists():
            z = np.load(f)
            out[key] = (z["V"], z["F"])
        else:
            missing[key] = d
    if missing:
        t0 = time.perf_counter()
        built: dict = {}
        workers = max(1, min(len(missing), os.cpu_count() or 1, 4))
        try:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for d, V, F in ex.map(_build_mesh, list(missing.values())):
                        built[_params_key(d)] = (V, F)
        except Exception:  # pragma: no cover - e.g. a sandbox without fork; fall back below
            built = {}
        for key, d in missing.items():
            VF = built.get(_params_key(d))
            if VF is None:
                _, V, F = _build_mesh(d)
                VF = (V, F)
            out[key] = VF
            if cache_dir is not None:
                np.savez_compressed(cache_dir / f"{_mesh_key(d)}.npz", V=VF[0], F=VF[1])
        print(f"\n[test_shrouded] built {len(missing)} meshes in {time.perf_counter() - t0:.1f} s "
              f"({workers} workers); {len(todo) - len(missing)} loaded from cache")
    for key, (V, F) in out.items():
        _seed_render3d_cache(todo[key], V, F)
    return out


@pytest.fixture(scope="module")
def meshes(mesh_cache_dir):
    """{(Z, beta2): (V, F)} for every grid combination (disk cache; built once otherwise)."""
    combos = sorted({(z, b) for z, b, _, _ in GRID})
    return load_or_build_meshes({(z, b): grid_params(z, b).to_dict() for z, b in combos},
                                mesh_cache_dir)


@pytest.fixture(scope="module")
def render_cache():
    return {}


def _render(meshes, render_cache, Z, beta2, preset_name, clutter, *, marker=False, seed=None,
            size=GRID_SIZE, **kw):
    key = (Z, beta2, preset_name, clutter, marker, seed, size, tuple(sorted(kw.items())))
    if key not in render_cache:
        preset, extra = PRESET_SPECS[preset_name] if preset_name in PRESET_SPECS else (preset_name, {})
        p = grid_params(Z, beta2)
        img, truth = render3d.render_params(
            p, preset, size=(size, size), mesh=meshes[(Z, beta2)], clutter=clutter,
            noise_sigma=4.0 if clutter else 0.0, marker=marker,
            seed=(Z * 100 + int(beta2)) if seed is None else seed, **extra, **kw)
        render_cache[key] = (img, truth, p)
    return render_cache[key]


# ---------------------------------------------------------------------------
# result helpers
# ---------------------------------------------------------------------------
def _prov(res, name):
    p = res.provenance[name]
    if isinstance(p, dict):
        return p["value"], p["source"], float(p["confidence"])
    return p.value, p.source, float(p.confidence)


def _check_result_structure(res, img):
    assert isinstance(res.params, ImpellerParams)
    res.params.validate()
    assert isinstance(res.provenance, dict) and res.provenance
    for name in ("outer_diameter", "inner_diameter", "hub_diameter", "blade_count",
                 "blade_angle_inlet", "blade_angle_outlet", "blade_width"):
        assert name in res.provenance, f"no provenance for {name}"
        v, src, conf = _prov(res, name)
        assert src in PROVENANCE_SOURCES, (name, src)
        assert 0.0 <= conf <= 1.0, (name, conf)
        # params are rounded (3 decimals) by the analyzer; provenance holds the raw value
        assert v == pytest.approx(getattr(res.params, name), rel=1e-3, abs=0.01), name
    assert res.rim is not None
    assert hasattr(res.rim, "center") and hasattr(res.rim, "axes")
    assert isinstance(res.rectified, np.ndarray) and res.rectified.ndim == 2
    assert isinstance(res.debug, dict)
    over = shrouded.draw_overlay(img, res)
    assert over.ndim == 3 and over.shape[:2] == img.shape[:2] and over.dtype == np.uint8


RESULTS: list[dict] = []


@pytest.fixture(scope="module", autouse=True)
def _summary():
    yield
    if not RESULTS:
        return
    print("\n[test_shrouded] worst errors per preset over the grid:")
    for name in PRESET_SPECS:
        rows = [r for r in RESULTS if r["preset"] == name]
        if not rows:
            continue
        print(f"  {name:12s} n={len(rows):2d}  Z ok {sum(r['z_ok'] for r in rows)}/{len(rows)}  "
              f"D_eye {max(r['d_eye'] for r in rows) * 100:.1f} %  hub {max(r['hub'] for r in rows) * 100:.1f} %  "
              f"beta1 {max(r['beta1'] for r in rows):.1f} deg  beta2 {max(r['beta2'] for r in rows):.1f} deg  "
              f"vis {min(r['vis'] for r in rows):.2f}..{max(r['vis'] for r in rows):.2f}  "
              f"t {max(r['t'] for r in rows):.1f} s")


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Z,beta2,preset,clutter", GRID, ids=GRID_IDS)
def test_grid(meshes, render_cache, Z, beta2, preset, clutter):
    img, truth, p = _render(meshes, render_cache, Z, beta2, preset, clutter)
    t0 = time.perf_counter()
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter,
                                    blade_width_mm=p.b2, blade_width_inlet_mm=p.b1)
    dt = time.perf_counter() - t0
    est = res.params
    rec = {"preset": preset, "Z": Z, "beta2": beta2, "clutter": clutter, "t": dt,
           "z_ok": est.blade_count == Z,
           "d_eye": abs(est.r_eye * 2 - p.r_eye * 2) / (p.r_eye * 2),
           "hub": abs(est.hub_diameter - p.hub_diameter) / p.hub_diameter,
           "beta1": abs(est.blade_angle_inlet - p.blade_angle_inlet),
           "beta2": abs(est.blade_angle_outlet - p.blade_angle_outlet),
           "vis": truth["visible_blade_fraction"]}
    RESULTS.append(rec)
    _check_result_structure(res, img)
    assert dt < ANALYZE_TIME_BOUND_S, f"analyze_shrouded took {dt:.1f} s"
    assert est.blade_count == Z, f"Z: truth {Z} est {est.blade_count}"
    assert _prov(res, "outer_diameter")[1] == "user"
    assert est.outer_diameter == pytest.approx(p.outer_diameter, rel=D_TOL)
    assert est.eye_diameter is not None
    assert rec["d_eye"] <= D_TOL, f"D_eye: truth {p.r_eye * 2} est {est.eye_diameter}"
    assert rec["hub"] <= D_TOL, f"hub: truth {p.hub_diameter} est {est.hub_diameter}"
    assert rec["beta1"] <= BETA1_TOL_DEG, \
        f"beta1: truth {p.blade_angle_inlet} est {est.blade_angle_inlet}"
    assert _prov(res, "blade_angle_outlet")[1] == "extrapolated"
    if preset == "axial_close":
        assert rec["beta2"] <= BETA2_EXTRAP_TOL_DEG, \
            f"beta2 (extrapolated): truth {beta2} est {est.blade_angle_outlet}"
    # blade widths were given -> 'user'
    assert est.blade_width == pytest.approx(p.b2) and _prov(res, "blade_width")[1] == "user"
    assert est.b1 == pytest.approx(p.b1)
    # rim / eye ellipses sit where the renderer put them (original image coordinates)
    tr = truth["rim_ellipse"]
    assert math.hypot(res.rim.center[0] - tr["center"][0], res.rim.center[1] - tr["center"][1]) < 6.0
    assert abs(res.rim.axes[0] - tr["axes"][0]) < 6.0


# ---------------------------------------------------------------------------
# scale fallbacks (the user's "D2 / b unknown" cases)
# ---------------------------------------------------------------------------
def test_prior_scale_keeps_ratios(meshes, render_cache):
    """No D2, no D_eye, no reference disc: D2 := 300 ('prior', low confidence) and all the
    diameters are reported consistently with it (ratios still measured).  A far camera is
    used because without any known length the eye/rim perspective magnification cannot be
    corrected (SPEC2 step 2: 'assume a far camera and add a warning'; with axial_close the eye
    ratio is biased by ~13 %)."""
    img, truth, p = _render(meshes, render_cache, 11, 40.0, "axial_far", True)
    res = shrouded.analyze_shrouded(img)
    est = res.params
    v, src, conf = _prov(res, "outer_diameter")
    assert src == "prior" and conf <= 0.1
    assert est.outer_diameter == pytest.approx(300.0)
    assert est.blade_count == 11
    assert abs(est.r_eye / est.r2 - p.r_eye / p.r2) <= RATIO_TOL * (p.r_eye / p.r2)
    assert abs(est.hub_diameter / est.outer_diameter - p.hub_diameter / p.outer_diameter) \
        <= RATIO_TOL * (p.hub_diameter / p.outer_diameter)
    # blade width unknown -> prior 0.2 * D2 with a warning
    v, src, conf = _prov(res, "blade_width")
    assert src == "prior" and est.blade_width == pytest.approx(0.2 * est.outer_diameter)
    assert conf <= 0.3
    assert isinstance(est.meta.get("warnings", []), list)


def test_eye_diameter_scale_far_camera(meshes, render_cache):
    """Only D_eye known, far camera: D2 within 4 %."""
    img, truth, p = _render(meshes, render_cache, 11, 40.0, "axial_far", False)
    res = shrouded.analyze_shrouded(img, eye_diameter_mm=p.r_eye * 2)
    est = res.params
    assert est.eye_diameter == pytest.approx(p.r_eye * 2, rel=1e-6)
    if "eye_diameter" in res.provenance:
        assert _prov(res, "eye_diameter")[1] == "user"
    assert est.outer_diameter == pytest.approx(p.outer_diameter, rel=D_TOL)
    assert est.blade_count == 11


def test_both_diameters_perspective(meshes, render_cache):
    """D2 and D_eye given with a close camera: the perspective correction runs, hub within 4 %
    and the camera-distance estimate is reported in debug (when the analyzer exposes it)."""
    img, truth, p = _render(meshes, render_cache, 11, 40.0, "axial_close", False)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.r_eye * 2)
    est = res.params
    assert est.blade_count == 11
    assert abs(est.hub_diameter - p.hub_diameter) / p.hub_diameter <= D_TOL
    cam = res.debug.get("camera_distance_mm") or res.debug.get("camera_distance")
    if cam is not None:
        # truth: camera 1.2*D2 above the backplate plane; rim plane at tb + b2
        d_truth = truth["camera"]["position"][2] - (p.backplate_thickness + p.b2)
        assert abs(float(cam) - d_truth) / d_truth < 0.25, (cam, d_truth)


# ---------------------------------------------------------------------------
# wrap-angle beta2 from marker side views
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Z,beta2", [(11, 40.0), (9, 55.0)])
def test_wrap_angle_beta2(meshes, render_cache, Z, beta2):
    """Axial image + 3 'side' views at camera azimuths 240/270/300 with a bright marker on the
    shroud rim -> beta2 'wrap' within 3 deg."""
    img, truth, p = _render(meshes, render_cache, Z, beta2, "axial_close", False, marker=True)
    assert truth["marker"]["visible_pixels"] > 0
    rim_images = []
    for az in (240.0, 270.0, 300.0):
        rim_img, rim_truth, _ = _render(meshes, render_cache, Z, beta2, "side", False, marker=True,
                                        size=1200, azimuth_deg=az)
        assert rim_truth["marker"]["visible_pixels"] > 0
        rim_images.append(rim_img)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, blade_width_mm=p.b2,
                                    blade_width_inlet_mm=p.b1, rim_images=rim_images,
                                    marker_hint="bright")
    est = res.params
    assert est.blade_count == Z
    v, src, conf = _prov(res, "blade_angle_outlet")
    assert src == "wrap", f"beta2 provenance {src}, expected 'wrap'"
    assert conf >= 0.5
    assert abs(est.blade_angle_outlet - beta2) <= BETA2_WRAP_TOL_DEG, \
        f"beta2 (wrap): truth {beta2} est {est.blade_angle_outlet}"
    # b2 from the rim gap is possible too; when the user gives it, it stays 'user'
    assert _prov(res, "blade_width")[1] == "user"


def test_rim_gap_gives_blade_width(meshes, render_cache):
    """Without a user value the blade width comes from the rim image (gap height / rim
    diameter * D2) rather than the prior."""
    Z, beta2 = 11, 40.0
    img, truth, p = _render(meshes, render_cache, Z, beta2, "axial_close", False, marker=True)
    rim_img, _, _ = _render(meshes, render_cache, Z, beta2, "side", False, marker=True, size=1200,
                            azimuth_deg=270.0)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, rim_images=[rim_img],
                                    marker_hint="bright")
    v, src, conf = _prov(res, "blade_width")
    assert src == "measured", src
    assert abs(res.params.blade_width - p.b2) / p.b2 <= 0.15


# ---------------------------------------------------------------------------
# factory params, real sample, hints
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def factory_mesh(mesh_cache_dir):
    return load_or_build_meshes({"factory": SHROUDED_PARAMS_FACTORY().to_dict()}, mesh_cache_dir)["factory"]


def test_factory_params_axial_close(factory_mesh):
    """SPEC2 default params (LE just outside the eye): Z exact and D_eye/hub within 4 % from the
    close view, where ~34 % of the blade span is visible through the eye."""
    p = SHROUDED_PARAMS_FACTORY()
    img, truth = render3d.render_params(p, "axial_close", size=(1200, 1200), mesh=factory_mesh,
                                        clutter=True, noise_sigma=4.0, seed=7)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, blade_width_mm=p.b2,
                                    blade_width_inlet_mm=p.b1)
    est = res.params
    _check_result_structure(res, img)
    assert est.blade_count == p.blade_count
    # D2 only + close camera: the eye ring (35 mm nearer the camera than the rim) is magnified
    # ~13 % and SPEC2 says to assume a far camera then; measured +5.9 % -> 8 % here
    assert abs(est.r_eye * 2 - p.r_eye * 2) / (p.r_eye * 2) <= 0.08
    assert abs(est.hub_diameter - p.hub_diameter) / p.hub_diameter <= D_TOL
    # the LE is hidden under the lip: r1 falls back to r_eye with provenance 'prior'
    v, src, conf = _prov(res, "inner_diameter")
    assert src in ("prior", "measured")
    if src == "prior":
        assert est.inner_diameter == pytest.approx(est.r_eye * 2, rel=0.05)


def test_blade_count_hint_breaks_ties(meshes, render_cache):
    img, truth, p = _render(meshes, render_cache, 13, 55.0, "axial_far", True)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, blade_count_hint=13)
    assert res.params.blade_count == 13
    v, src, conf = _prov(res, "blade_count")
    assert src in ("measured", "user")


def test_real_axial_sample_runs():
    """No ground truth: the pipeline must run on the real phone photo, find the rim and the
    inlet ring, count 10..12 blades (11 by eye) and report provenance for everything."""
    img = cv2.imread(str(REAL_AXIAL), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    t0 = time.perf_counter()
    res = shrouded.analyze_shrouded(img)
    dt = time.perf_counter() - t0
    assert dt < 3 * ANALYZE_TIME_BOUND_S
    _check_result_structure(res, img)
    assert res.eye is not None
    assert 10 <= res.params.blade_count <= 12, res.params.blade_count
    v, src, conf = _prov(res, "outer_diameter")
    assert src == "prior" and v == pytest.approx(300.0)
    assert "expected_from_eye" in res.debug
    assert 1.0 <= res.params.blade_angle_inlet <= 179.0
    assert 1.0 <= res.params.blade_angle_outlet <= 179.0
    # with D2 given the result scales consistently
    res2 = shrouded.analyze_shrouded(img, outer_diameter_mm=320.0)
    assert _prov(res2, "outer_diameter")[1] == "user"
    assert res2.params.r_eye / res2.params.r2 == pytest.approx(res.params.r_eye / res.params.r2, rel=0.02)
    assert res2.params.blade_count == res.params.blade_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_analyze_shrouded_and_auto(tmp_path, meshes, render_cache, capsys):
    from fanscan import cli

    img, truth, p = _render(meshes, render_cache, 11, 40.0, "axial_close", True)
    png = tmp_path / "shot.png"
    render3d.save(png, img)
    out = tmp_path / "params.json"
    overlay = tmp_path / "overlay.png"
    rc = cli.main(["analyze", str(png), "--mode", "shrouded", "--outer-diameter-mm", "300",
                   "--blade-width-mm", "55", "--blade-width-inlet-mm", "75",
                   "--out", str(out), "--overlay", str(overlay)])
    assert rc == 0
    est = ImpellerParams.load(out)
    assert est.blade_count == 11 and est.outer_diameter == 300.0 and est.blade_width == 55.0
    assert est.blade_width_inlet == 75.0
    assert overlay.exists() and cv2.imread(str(overlay)) is not None
    text = capsys.readouterr().out
    summary = json.loads(text[:text.index("\nparameter")])
    assert summary["mode"] == "shrouded" and summary["provenance"]["outer_diameter"]["source"] == "user"
    assert "blade_angle_outlet" in text and "extrapolated" in text
    assert cli.provenance_table(shrouded.analyze_shrouded(img)).count("prior") >= 1
    # auto mode picks shrouded (rim + eye found, no reference disc) and needs no scale
    rc = cli.main(["analyze", str(png), "--out", str(tmp_path / "auto.json")])
    assert rc == 0
    auto = json.loads(capsys.readouterr().out.split("\nparameter")[0])
    assert auto["mode"] == "shrouded" and auto["provenance"]["outer_diameter"]["source"] == "prior"


def test_cli_pipeline3d(tmp_path, meshes, capsys):
    """render -> analyze_shrouded -> build with marker side views: the table reports beta2 as
    pass/fail (wrap) and the run passes the v0.1 rules.  Builds a STEP (~15 s)."""
    from fanscan import cli

    p = grid_params(11, 40.0)
    pj = tmp_path / "truth.json"
    p.save(pj)
    assert (11, 40.0) in meshes            # the fixture seeded render3d's cache for the CLI
    out_dir = tmp_path / "pipe"
    rc = cli.main(["pipeline3d", "--out-dir", str(out_dir), "--preset", "axial_close",
                   "--params", str(pj), "--size", "1000", "--clutter",
                   "--rim-azimuths", "240,270,300"])
    text = capsys.readouterr().out
    for name in ("render.png", "render.truth.json", "rim_az240.png", "rim_az270.png",
                 "rim_az300.png", "params.json", "overlay.png", "provenance.json", "impeller.step"):
        assert (out_dir / name).exists(), name
    assert (out_dir / "impeller.step").read_bytes()[:12] == b"ISO-10303-21"
    prov = json.loads((out_dir / "provenance.json").read_text())
    assert set(prov) >= {"outer_diameter", "blade_count", "blade_angle_outlet", "blade_width"}
    assert prov["outer_diameter"]["source"] == "user" and prov["blade_width"]["source"] == "user"
    # plumbing: the table's beta2 status follows the provenance (pass/fail only for 'wrap')
    assert "blade_angle_outlet" in text and ("PASS" in text or "FAIL" in text)
    assert "parameter" in text and "source" in text          # provenance table printed
    if prov["blade_angle_outlet"]["source"] == "wrap":
        assert "info (extrapolated" not in text
    else:
        assert "info (extrapolated" in text
    assert rc == (1 if "FAIL:" in text else 0)
    # accuracy gate (SPEC2: rim images given -> beta2 'wrap' and the v0.1 rules pass)
    assert prov["blade_angle_outlet"]["source"] == "wrap", prov["blade_angle_outlet"]
    assert rc == 0, text
    # the prior path: no D2 -> diameters are info, exit 0 regardless of the absolute scale
    rc = cli.main(["pipeline3d", "--out-dir", str(tmp_path / "pipe2"), "--preset", "axial_far",
                   "--params", str(pj), "--size", "800", "--no-outer-diameter", "--no-blade-width",
                   "--skip-build"])
    text = capsys.readouterr().out
    assert "info (prior)" in text
    assert rc == 0, text
