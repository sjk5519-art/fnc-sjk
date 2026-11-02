"""Tests for fanscan.render3d (software renderer, camera presets, truth dicts) and the
``render`` CLI subcommand.

Achieved numbers (SHROUDED_PARAMS_FACTORY, 1200x1200, 17.7k triangles, this machine):
  mesh build + tessellation (cadquery)   ~15 s, done once per process (render3d._MESH_CACHE)
  render total per preset                1.9-2.6 s (SPEC2 bound: < 15 s for 20k triangles)
  axial presets: truth rim ellipse vs cv2.fitEllipse of the rendered silhouette  <= 0.5 px
  tilt preset: truth shroud-rim ellipse (far half) vs rendered silhouette edge   <= 1 px
Timing of the whole file: ~40 s (the mesh build dominates).
"""
import json
import math
import time

import cv2
import numpy as np
import pytest

from fanscan import render3d
from fanscan.params import SHROUDED_PARAMS_FACTORY

SIZE = (1200, 1200)
RENDER_TIME_BOUND_S = 15.0          # SPEC2: 20k triangles at 1200x1200 in < 15 s
RIM_TOL_PX = 2.0                    # SPEC2: truth rim ellipse vs rendered rim edge within 2 px


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def params():
    return SHROUDED_PARAMS_FACTORY()


@pytest.fixture(scope="module")
def mesh(params):
    t0 = time.perf_counter()
    V, F = render3d.mesh_from_params(params)
    dt = time.perf_counter() - t0
    assert dt < 90.0, f"mesh build took {dt:.1f} s"
    return V, F


@pytest.fixture(scope="module")
def renders(params, mesh):
    """{preset: (img, truth, wall_time_s)} for all five presets (no marker, no clutter)."""
    out = {}
    for name in render3d.PRESETS:
        t0 = time.perf_counter()
        img, truth = render3d.render_params(params, name, size=SIZE, mesh=mesh)
        out[name] = (img, truth, time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _silhouette(truth) -> np.ndarray:
    """Boolean mask of pixels covered by the part (from the id image)."""
    return truth["_id_image"] >= 0


def _outer_contour(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert cs, "no silhouette found"
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the nearest silhouette boundary pixel."""
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    boundary = (m > 0) & (er == 0)
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


def _ellipse_points(e: dict, n: int = 720) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = e["center"]
    a, b = e["axes"]
    th = math.radians(e["angle_deg"])
    x = a * np.cos(t)
    y = b * np.sin(t)
    return np.stack([cx + x * math.cos(th) - y * math.sin(th),
                     cy + x * math.sin(th) + y * math.cos(th)], axis=1)


# ---------------------------------------------------------------------------
# mesh / camera basics
# ---------------------------------------------------------------------------
def test_mesh_is_sane(mesh):
    V, F = mesh
    assert V.ndim == 2 and V.shape[1] == 3 and np.isfinite(V).all()
    assert F.ndim == 2 and F.shape[1] == 3 and F.dtype.kind == "i"
    assert F.min() >= 0 and F.max() < len(V)
    assert 5_000 <= len(F) <= 60_000, len(F)
    r = np.hypot(V[:, 0], V[:, 1])
    assert r.max() == pytest.approx(SHROUDED_PARAMS_FACTORY().r2 + render3d.BACKPLATE_RIM_MM, abs=0.5)
    assert V[:, 2].min() == pytest.approx(0.0, abs=1e-6)


def test_camera_matrices_reproduce_projection(params):
    cam = render3d.camera_preset("oblique", params, SIZE)
    K, Rt = cam.intrinsics(), cam.extrinsics()
    assert K.shape == (3, 3) and Rt.shape == (3, 4)
    rng = np.random.default_rng(0)
    P = rng.uniform(-150, 150, (200, 3)) + [0, 0, 40]
    hom = (K @ (Rt @ np.hstack([P, np.ones((200, 1))]).T)).T
    px = hom[:, :2] / hom[:, 2:3]
    proj = cam.project(P)
    assert np.allclose(px, proj[:, :2], atol=1e-6)
    assert np.allclose(hom[:, 2], proj[:, 2], atol=1e-6)
    # rotation is orthonormal with the forward axis pointing at the target
    R = cam.rotation()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    f = (cam.target - cam.position) / np.linalg.norm(cam.target - cam.position)
    assert np.allclose(R[2], f, atol=1e-9)


@pytest.mark.parametrize("name", render3d.PRESETS)
def test_camera_preset_frames_the_part(params, name):
    cam = render3d.camera_preset(name, params, SIZE)
    assert cam.size == SIZE and cam.focal_px > 0
    # the impeller axis at the backplate projects inside the frame for every preset
    c = cam.project(np.array([[0.0, 0.0, params.backplate_thickness]]))[0]
    assert 0 <= c[0] < SIZE[0] and 0 <= c[1] < SIZE[1] and c[2] > 0


# ---------------------------------------------------------------------------
# rendering all presets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", render3d.PRESETS)
def test_preset_renders_and_truth_is_serialisable(renders, name):
    img, truth, _ = renders[name]
    H, W = SIZE[1], SIZE[0]
    assert img.dtype == np.uint8 and img.shape == (H, W)
    part = _silhouette(truth)
    frac = part.mean()
    assert 0.05 < frac < 0.95, f"{name}: part covers {frac:.2f} of the frame"
    # part is lighter than the (dark) default background on average
    assert img[part].mean() > img[~part].mean() + 40
    # truth dict content
    assert truth["preset"] == name
    assert set(truth["part_pixel_counts"]) == set(render3d.PART_IDS)
    assert truth["params"]["outer_diameter"] == SHROUDED_PARAMS_FACTORY().outer_diameter
    assert truth["px_per_mm"] > 0 and np.isfinite(truth["px_per_mm"])
    js = json.dumps(render3d.truth_json(truth))
    back = json.loads(js)
    assert "_id_image" not in back and "_depth" not in back
    assert back["camera"]["K"][0][0] == pytest.approx(truth["camera"]["K"][0][0])


@pytest.mark.parametrize("name", render3d.PRESETS)
def test_render_time_bound(renders, name):
    _, truth, wall = renders[name]
    tm = truth["timing_s"]
    assert tm["total"] < RENDER_TIME_BOUND_S, f"{name}: render took {tm['total']:.2f} s"
    assert wall < RENDER_TIME_BOUND_S + 5.0, f"{name}: wall time {wall:.2f} s"
    assert truth["n_triangles"] >= 5_000


@pytest.mark.parametrize("name", ("axial_far", "axial_close"))
def test_axial_rim_ellipse_matches_silhouette(renders, name):
    """Truth rim ellipse == cv2.fitEllipse of the rendered outer silhouette (<= 2 px)."""
    _, truth, _ = renders[name]
    rim = truth["rim_ellipse"]
    assert rim is not None and truth["rim_source"] == "shroud"
    pts = _outer_contour(_silhouette(truth))
    (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
    a, b = max(w, h) / 2, min(w, h) / 2
    assert abs(cx - rim["center"][0]) <= RIM_TOL_PX and abs(cy - rim["center"][1]) <= RIM_TOL_PX
    assert abs(a - rim["axes"][0]) <= RIM_TOL_PX and abs(b - rim["axes"][1]) <= RIM_TOL_PX
    # on axis the rim is a circle centred on the projected axis
    assert rim["axes"][1] / rim["axes"][0] > 0.995
    assert np.hypot(*(np.array(rim["center"]) - np.array(truth["center_xy"]))) < 1.0
    # px_per_mm at the rim plane reproduces the rim radius
    assert truth["px_per_mm_rim"] * SHROUDED_PARAMS_FACTORY().r2 == pytest.approx(rim["axes"][0], abs=1.0)


@pytest.mark.parametrize("tilt_deg", (10.0, 15.0))
def test_tilt_rim_edge_matches_truth(params, mesh, tilt_deg):
    """In a tilted view the outer silhouette is the union of the shroud rim (far side) and
    the backplate / rim underside (near side).  Check (a) every far-half point of the truth
    shroud-rim ellipse lies within 2 px of the rendered silhouette edge and (b) the whole
    silhouette edge lies within 2 px of the union of the three truth ellipses."""
    img, truth = render3d.render_params(params, "tilt", size=SIZE, mesh=mesh, tilt_deg=tilt_deg)
    sil = _silhouette(truth)
    dist = _boundary_distance(sil)
    rim = truth["rim_shroud_ellipse"]
    assert rim is not None
    # foreshortening of the rim matches the tilt (perspective adds a fraction of a degree)
    tilt_est = math.degrees(math.acos(rim["axes"][1] / rim["axes"][0]))
    assert abs(tilt_est - tilt_deg) < 1.5, tilt_est
    # (a) far half = image rows above the rim centre (camera displaced towards -y, +y up on screen)
    pts = _ellipse_points(rim)
    far = pts[pts[:, 1] < rim["center"][1] - 0.2 * rim["axes"][1]]
    assert len(far) > 100
    xi = np.clip(np.rint(far[:, 0]).astype(int), 0, SIZE[0] - 1)
    yi = np.clip(np.rint(far[:, 1]).astype(int), 0, SIZE[1] - 1)
    d = dist[yi, xi]
    assert np.percentile(d, 99) <= RIM_TOL_PX, f"far rim edge off by {np.percentile(d, 99):.2f} px"
    # (b) silhouette edge vs union of truth ellipses
    union = np.zeros_like(sil)
    for key in ("rim_shroud_ellipse", "rim_underside_ellipse", "backplate_ellipse"):
        union |= render3d.ellipse_mask(truth[key], SIZE)
    dist_u = _boundary_distance(union)
    edge = _outer_contour(sil)
    du = dist_u[np.clip(np.rint(edge[:, 1]).astype(int), 0, SIZE[1] - 1),
                np.clip(np.rint(edge[:, 0]).astype(int), 0, SIZE[0] - 1)]
    assert np.percentile(du, 99) <= RIM_TOL_PX, f"silhouette off by {np.percentile(du, 99):.2f} px"


def test_visible_blade_fraction_and_ids(renders, params):
    close, far, obl = renders["axial_close"][1], renders["axial_far"][1], renders["oblique"][1]
    # the closer the camera, the further out the blade roots are seen through the eye
    assert 0.0 <= far["visible_blade_fraction"] <= close["visible_blade_fraction"] <= 1.0
    assert close["visible_blade_fraction"] > 0.1
    assert close["r_vis_root_mm"] > params.r_eye
    # SPEC2 remark: with the factory LE outside the eye nothing is visible from far away
    assert far["visible_blade_fraction"] == pytest.approx(0.0, abs=0.05)
    # blade pixels exist inside the eye for the close view, and blades dominate the oblique view
    ids = close["_id_image"]
    eye_mask = render3d.ellipse_mask(close["eye_ellipse"], SIZE)
    assert np.sum((ids == render3d.PART_IDS["blade"]) & eye_mask) > 500
    assert obl["part_pixel_counts"]["blade"] > 10_000
    assert obl["part_pixel_counts"]["shroud"] > 10_000


def test_eye_and_bore_ellipses(renders, params):
    _, truth, _ = renders["axial_close"]
    eye, bore, hub = truth["eye_ellipse"], truth["bore_ellipse"], truth["hub_ellipse"]
    assert eye["r_mm"] == params.r_eye and bore["r_mm"] == params.hub_bore_diameter / 2
    assert bore["axes"][0] < hub["axes"][0] < eye["axes"][0] < truth["rim_ellipse"]["axes"][0]
    # the eye ring is closer to the camera than the rim -> magnified relative to its radius
    assert eye["axes"][0] / truth["rim_ellipse"]["axes"][0] > params.r_eye / params.r2 * 1.05
    # the bore shows the background: its centre pixel is background in the id image
    cx, cy = map(int, map(round, bore["center"]))
    assert truth["_id_image"][cy, cx] == -1


@pytest.mark.parametrize("azimuth", (240.0, 270.0, 300.0))
def test_marker_visible_in_side_views(params, mesh, azimuth):
    img, truth = render3d.render_params(params, "side", size=SIZE, mesh=mesh, marker=True,
                                        azimuth_deg=azimuth)
    m = truth["marker"]
    assert m is not None and m["visible_pixels"] > 100
    px, py = m["px"]
    assert 0 <= px < SIZE[0] and 0 <= py < SIZE[1]
    # the marker is the only saturated blob: its centroid matches the projected marker centre
    sat = (img >= 254).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(sat)
    assert n >= 2
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # the marker is a cube seen obliquely: the centroid of its lit faces is within ~one
    # marker size (2.5 % of D2) of the projected cube centre (measured: 7 px at az 240)
    marker_px = 0.025 * params.outer_diameter * truth["px_per_mm_rim"]
    assert np.hypot(cents[k][0] - px, cents[k][1] - py) < 0.75 * marker_px
    # marker azimuth relative to the camera: 270 is at the image centre, 240/300 off to the sides
    off = px - (SIZE[0] - 1) / 2
    if azimuth == 270.0:
        assert abs(off) < 5.0
    else:
        assert abs(off) > 150.0
    # the rim gap (blade trailing edges between backplate and shroud) is in view
    assert truth["part_pixel_counts"]["blade"] > 2_000


def test_clutter_background_is_deterministic():
    a = render3d.clutter_background((400, 300), seed=3)
    b = render3d.clutter_background((400, 300), seed=3)
    c = render3d.clutter_background((400, 300), seed=4)
    assert a.dtype == np.uint8 and a.shape == (300, 400)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.std() > 15 and 60 < a.mean() < 220


def test_render_with_clutter_noise_blur(params, mesh):
    img, truth = render3d.render_params(params, "axial_close", size=(600, 600), mesh=mesh,
                                        clutter=True, noise_sigma=6.0, blur_sigma=1.0, seed=1)
    assert img.shape == (600, 600) and truth["clutter"] is True
    bg = truth["_id_image"] < 0
    assert bg.any() and img[bg].std() > 10          # textured background, not flat


def test_render_low_level_api(params, mesh):
    V, F = mesh
    cam = render3d.camera_preset("axial_far", params, (300, 300))
    out = render3d.render(V, F, cam, return_depth=True, face_ids=render3d.classify_faces(V, F, params))
    img, depth, ids = out
    assert img.shape == (300, 300) and depth.shape == (300, 300) and ids.shape == (300, 300)
    hit = ids >= 0
    assert hit.any() and np.isfinite(depth[hit]).all() and np.isinf(depth[~hit]).all()
    assert set(np.unique(ids[hit])).issubset(set(render3d.PART_IDS.values()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_render(tmp_path, params, mesh, capsys):
    from fanscan import cli

    pj = tmp_path / "p.json"
    params.save(pj)
    out = tmp_path / "r" / "axial.png"
    rc = cli.main(["render", "--preset", "axial_close", "--out", str(out), "--params", str(pj),
                   "--size", "500", "--clutter", "--noise", "3", "--blur", "0.5", "--marker"])
    assert rc == 0
    assert out.exists()
    truth = json.loads(out.with_suffix(".truth.json").read_text())
    assert truth["preset"] == "axial_close" and truth["size"] == [500, 500]
    assert truth["marker"]["visible_pixels"] > 0
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (500, 500)
    text = capsys.readouterr().out
    assert "preset axial_close" in text and "visible blade fraction" in text
    # tilt preset with --tilt-deg
    out2 = tmp_path / "tilt.png"
    rc = cli.main(["render", "--preset", "tilt", "--tilt-deg", "10", "--out", str(out2),
                   "--params", str(pj), "--size", "400"])
    assert rc == 0
    t2 = json.loads(out2.with_suffix(".truth.json").read_text())
    e = t2["rim_shroud_ellipse"]
    assert abs(math.degrees(math.acos(e["axes"][1] / e["axes"][0])) - 10.0) < 1.5
    with pytest.raises(SystemExit):
        cli.main(["render", "--preset", "nope", "--out", str(tmp_path / "x.png"), "--params", str(pj)])
