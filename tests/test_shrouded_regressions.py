"""Regression tests for the 17 CONFIRMED review findings in ``fanscan/shrouded.py``.

Cheap cases only: the real photo (``samples/real_axial_eye.jpg``, 8 blades, clean dark elliptical
bore in the middle, blades sweep clockwise seen from the inlet) is analysed once per variant in a
module fixture - at 900 px wherever the finding does not depend on the resolution (2 s instead of
5 s) - plus a few cached synthetic renders (``render_cache``) for the wrap-angle findings.

Finding -> test
  0  axis runs to its search boundary / lands 37 mm off the bore   test_axis_is_anchored_on_the_bore,
                                                                   test_centre_correction_refuses_clipped_fit
  1  beta1 'measured' from one ring-hugging trace                   test_beta1_is_not_a_confident_measurement_on_the_real_photo,
                                                                   test_traces_on_ring_or_hub_are_rejected
  2  handedness flips under a downscale                            test_handedness_is_stable_and_flips_under_mirror,
                                                                   test_undetermined_handedness_is_reported
  3  hub flange lost under gamma 0.8                                test_hub_survives_gamma
  4  wrap path LE phase evaluated at r_a                           test_wrap_le_phase_bug_fixed
  5  wrap with one known length overrides a better value           test_wrap_with_one_known_length_keeps_extrapolated
  6  featureless eye -> Z = 4 'measured'                            test_featureless_eye_is_not_a_blade_count
  7  blade_count_hint 0 / 1                                         test_blade_count_hint_validation
  8  unreadable rim image                                           test_missing_rim_image
  9  rim cut by the frame                                           test_rim_cut_by_frame_is_flagged
  10 / 14 clamped eye diameter keeps pre-clamp provenance           test_provenance_follows_clamps
  11 / 16 float [0, 1] / uint16 clipped                             test_dtype_aware_gray
  12 invalid handedness string                                      test_invalid_handedness
  13 tiny rim image                                                 test_tiny_rim_image
  15 clamped D1 keeps pre-clamp provenance                          test_provenance_follows_clamps
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fanscan import shrouded
from fanscan.params import ImpellerParams

import test_shrouded as T                      # helpers of the main shrouded test module

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "samples" / "real_axial_eye.jpg"
PROV_FIELDS = ("outer_diameter", "eye_diameter", "inner_diameter", "hub_diameter", "hub_bore_diameter",
               "blade_angle_inlet", "blade_angle_outlet", "blade_width", "blade_width_inlet",
               "blade_thickness", "shroud_cone_height", "shroud_inlet_height")


# ---------------------------------------------------------------------------
# fixtures: the real photo and its perturbations
# ---------------------------------------------------------------------------
def _gamma(img, g):
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return cv2.LUT(img, lut)


@pytest.fixture(scope="module")
def real_bgr():
    img = cv2.imread(str(REAL))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def real900(real_bgr):
    s = 900.0 / max(real_bgr.shape[:2])
    return cv2.resize(real_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def real_variants(real_bgr, real900):
    """name -> ShroudedResult for the stability checks (D2 = 300 mm given)."""
    imgs = {
        "orig": real_bgr,
        "down900": real900,
        "rot90": cv2.rotate(real900, cv2.ROTATE_90_CLOCKWISE),
        "rot180": cv2.rotate(real900, cv2.ROTATE_180),
        "mirror": cv2.flip(real900, 1),
        "gamma0.8": _gamma(real_bgr, 0.8),
        "gamma1.2": _gamma(real900, 1.2),
    }
    return {k: shrouded.analyze_shrouded(v, outer_diameter_mm=300.0) for k, v in imgs.items()}


def _bore_centre_rect(res):
    """Independent bore ellipse (Otsu inside the eye, largest dark blob, hull ellipse)."""
    rect, e = res.rectified, res.debug["eye_rect"]
    m = np.zeros(rect.shape, np.uint8)
    (cx, cy), (w, h), ang = e.to_cv()
    cv2.ellipse(m, (int(cx), int(cy)), (int(0.6 * w / 2), int(0.6 * h / 2)), ang, 0, 360, 255, -1)
    sm = cv2.GaussianBlur(rect, (0, 0), 3)
    t, _ = cv2.threshold(sm[m > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _c = cv2.connectedComponentsWithStats(((sm < t) & (m > 0)).astype(np.uint8), 8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnt = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    (bx, by), (bw, bh), _a = cv2.fitEllipse(cv2.convexHull(max(cnt, key=len)))
    return bx, by, 0.25 * (bw + bh)


# ---------------------------------------------------------------------------
# 0. axis anchored on the bore
# ---------------------------------------------------------------------------
def test_axis_is_anchored_on_the_bore(real_variants):
    for name, res in real_variants.items():
        d = res.debug
        a_rim = d["rim_rect"].a
        assert d["bore_fit"] is not None, name
        bx, by, r_bore = _bore_centre_rect(res)
        cx, cy = d["centre_rect"]
        off = math.hypot(cx - bx, cy - by)
        assert off <= 0.05 * a_rim, f"{name}: axis {off:.0f} px from the bore centre (0.05 a = {0.05 * a_rim:.0f})"
        # bore diameter from the blob ellipse: 65.8 mm in the review (73 mm from the old profile)
        assert res.params.hub_bore_diameter == pytest.approx(66.0, rel=0.06), (name, res.params.hub_bore_diameter)
        assert res.provenance["hub_bore_diameter"].source == "measured"
        assert d["axis"]["offset_from_bore_px"] <= 0.05 * a_rim + 1e-6


def test_centre_correction_refuses_clipped_fit():
    """A _centre_correction that hits its clip limit is a runaway fit and is refused; a small one
    is returned."""
    n, per = 6, 100
    dphi = 2 * math.pi / (n * per)
    ids = np.arange(n)
    phi = ids * per * dphi

    def ba_with(amp):
        # a centre error dy shifts blade i by dy * cos(phi_i) / r_ref (image frame); hand = +1
        return {"phase_resid": -amp * np.cos(phi), "per": per, "phase_row": 0.0, "dphi": dphi,
                "r_a": 70.0, "r_b": 100.0, "hand": 1, "blade_ids": ids}

    r_ref = 85.0 * 4.0
    assert shrouded._centre_correction(ba_with(0.5), 4.0) is None          # dy = 0.5 r_ref > 0.12 r_ref
    corr = shrouded._centre_correction(ba_with(0.02), 4.0)
    assert corr is not None and corr[1] == pytest.approx(0.02 * r_ref, rel=1e-6) and abs(corr[0]) < 1e-6


# ---------------------------------------------------------------------------
# 1. beta1 'measured' only from trustworthy traces
# ---------------------------------------------------------------------------
def test_beta1_is_not_a_confident_measurement_on_the_real_photo(real_variants):
    """The review's failure: 'measured' 30.5 deg (conf 0.26) from ONE trace that follows the inlet-ring
    edge; 45-69 deg away under an axis move / 90 deg rotation.  Now: either a measurement backed by
    n_good >= max(2, Z//3) agreeing blades, or an honest prior with conf <= 0.15 - and the same
    statement for every perturbation."""
    statuses = set()
    for name, res in real_variants.items():
        v, src, conf = T._prov(res, "blade_angle_inlet")
        ba = res.debug.get("blade_analysis", {})
        if src == "measured":
            assert ba["n_good"] >= max(2, res.params.blade_count // 3), name
            assert ba["beta_a_agree_frac"] >= 0.7, name
        else:
            assert src == "prior" and conf <= 0.15, (name, src, conf)
            assert any("not measured" in w for w in res.warnings), (name, res.warnings)
        statuses.add(src)
        assert T._prov(res, "blade_angle_outlet")[2] <= 0.15 or T._prov(res, "blade_angle_outlet")[1] != "extrapolated"
    assert len(statuses) == 1, f"beta1 status differs between perturbations: {statuses}"


def test_traces_on_ring_or_hub_are_rejected(real_variants):
    """Traces that spend most of their length on the inlet-ring edge are dropped (they were the
    only 'good' trace in the review)."""
    ba = real_variants["orig"].debug["blade_analysis"]
    assert ba["rejected"], "expected ring-hugging traces to be rejected on the real photo"
    assert all(reason in ("inlet ring edge", "hub flange") for _i, reason in ba["rejected"])
    rejected = {i for i, _r in ba["rejected"]}
    assert not rejected & {f["blade"] for f in ba["fits"]}


def test_gate_falls_back_to_prior_with_few_good_traces(monkeypatch, real900):
    """n_good below max(2, Z//3) -> beta1 / beta2 are priors with low confidence, plus a warning."""
    orig = shrouded._blade_analysis

    def wrapped(*a, **k):
        out = orig(*a, **k)
        if not k.get("coarse_only") and "n_good" in out:
            out["n_good"] = 1
        return out

    monkeypatch.setattr(shrouded, "_blade_analysis", wrapped)
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, handedness="cw")
    for k in ("blade_angle_inlet", "blade_angle_outlet"):
        v, src, conf = T._prov(res, k)
        assert src == "prior" and conf <= 0.15, (k, src, conf)
    assert any("only 1 of" in w and "need" in w for w in res.warnings), res.warnings


# ---------------------------------------------------------------------------
# 2. handedness
# ---------------------------------------------------------------------------
def test_handedness_is_stable_and_flips_under_mirror(real_variants):
    hands = {k: r.params.meta["handedness"] for k, r in real_variants.items()}
    for name in ("orig", "down900", "rot90", "rot180", "gamma0.8", "gamma1.2"):
        assert hands[name] == "cw", hands
    assert hands["mirror"] == "ccw", hands
    assert real_variants["orig"].params.meta["handedness_source"] in ("orientation", "orientation+matched filter")


def test_undetermined_handedness_is_reported(monkeypatch, real900):
    """Both cues indecisive -> 'undetermined' (warning, source 'prior'), beta1 is a prior."""
    monkeypatch.setattr(shrouded, "_hand_from_orientation", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr(shrouded, "_HAND_MF_MARGIN", 1e6)
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0)
    assert res.params.meta["handedness"] == "undetermined"
    assert res.params.meta["handedness_source"] == "prior"
    assert any("undetermined" in w for w in res.warnings)
    v, src, conf = T._prov(res, "blade_angle_inlet")
    assert src == "prior" and conf <= 0.15
    # a user value always wins
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, handedness="cw")
    assert res.params.meta["handedness"] == "cw" and res.params.meta["handedness_source"] == "user"


def test_hand_cue_on_a_synthetic_render(render_cache):
    """The CAD convention (ccw blades) is recovered from the edge orientation on a render."""
    img, truth, p = T._grid_render(render_cache, 11, 40.0, "axial_close")
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    assert res.params.meta["handedness"] == "ccw"
    assert res.debug["hand_info"]["orientation_score"] < -0.4
    mir = shrouded.analyze_shrouded(img[:, ::-1].copy(), outer_diameter_mm=p.outer_diameter,
                                    eye_diameter_mm=p.eye_diameter)
    assert mir.params.meta["handedness"] == "cw"


# ---------------------------------------------------------------------------
# 3. hub flange under gamma
# ---------------------------------------------------------------------------
def test_hub_survives_gamma(real_variants):
    ref = real_variants["orig"].params.hub_diameter
    for name in ("gamma0.8", "gamma1.2", "down900"):
        res = real_variants[name]
        assert T._prov(res, "hub_diameter")[1] == "measured", (name, res.warnings)
        assert res.params.hub_diameter == pytest.approx(ref, rel=0.05), name


# ---------------------------------------------------------------------------
# 4 / 5. wrap angle
# ---------------------------------------------------------------------------
def test_wrap_le_phase_bug_fixed(render_cache):
    """Z13 b30 (the old strict-xfail 'known bias': LE phase taken at r_a instead of r1, beta2 26.5)."""
    res, p, img = T._analyze_wrap(render_cache, 13, 30.0)
    v, src, conf = T._prov(res, "blade_angle_outlet")
    assert src == "wrap", (src, res.warnings)
    assert abs(res.params.blade_angle_outlet - 30.0) <= 2.0, res.params.blade_angle_outlet
    wrap_true = T.geometry.wrap_from_betas(p.r1, p.r2, p.blade_angle_inlet, p.blade_angle_outlet)
    # wrap 99.9 deg before the fix (truth 90.9); 94.7 after - same 5 deg bound as test_wrap_angle_beta2
    assert abs(res.debug["wrap"]["wrap_deg"] - wrap_true) <= 5.0


def test_wrap_with_one_known_length_keeps_extrapolated(render_cache):
    """D2 only (no eye diameter / focal length) at a close camera: the backplate scale is not
    corrected, so the wrap value is reported in debug + a warning but NOT used."""
    img, rim_images, p = T._wrap_images(render_cache, 11, 40.0)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, rim_images=rim_images,
                                    marker_hint="bright")
    v, src, conf = T._prov(res, "blade_angle_outlet")
    assert src == "extrapolated", (src, res.warnings)
    w = res.debug.get("wrap", {})
    assert "beta2_wrap" in w and "rejected" in w, w
    assert any("not used" in x and "wrap" in x for x in res.warnings), res.warnings
    # with two lengths the wrap value is used again
    res2 = shrouded.analyze_shrouded(img, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter,
                                     rim_images=rim_images, marker_hint="bright")
    assert T._prov(res2, "blade_angle_outlet")[1] == "wrap"


# ---------------------------------------------------------------------------
# 6. featureless eye
# ---------------------------------------------------------------------------
def _flatten_eye(img_bgr, eye, kind):
    (cx, cy), (w, h), ang = eye.to_cv()
    m = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(0.96 * w / 2), int(0.96 * h / 2)), ang, 0, 360, 255, -1)
    out = img_bgr.copy()
    if kind == "flat":
        med = np.median(img_bgr[m > 0], axis=0)
        noise = np.random.default_rng(0).normal(0, 3, img_bgr.shape)
        out[m > 0] = np.clip(med[None, None, :] + noise, 0, 255).astype(np.uint8)[m > 0]
    else:
        out[m > 0] = cv2.GaussianBlur(img_bgr, (0, 0), 25)[m > 0]
    return out


@pytest.mark.parametrize("kind", ["flat", "blur"])
def test_featureless_eye_is_not_a_blade_count(real900, real_variants, kind):
    """The 8x8 CLAHE tile grid used to put a 4-fold pattern into an eye without blades ->
    Z = 4 'measured' (conf 0.58-0.96).  Now the 'no blade pattern' branch fires."""
    img = _flatten_eye(real900, real_variants["down900"].eye, kind)
    res = shrouded.analyze_shrouded(img, outer_diameter_mm=300.0)
    v, src, conf = T._prov(res, "blade_count")
    assert not (src == "measured" and res.params.blade_count == 4), (src, conf)
    assert src == "prior" and conf <= 0.1, (src, conf)
    assert any("no blade pattern" in w for w in res.warnings)
    assert T._prov(res, "blade_angle_inlet")[1] == "prior"


def test_featureless_render_eye(render_cache):
    """Same on a realistic axial_close render with the eye painted flat grey 200."""
    img, truth, p = T._grid_render(render_cache, 11, 40.0, "axial_close")
    e = truth["eye_ellipse"]
    out = img.copy()
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(m, (int(e["center"][0]), int(e["center"][1])), (int(0.96 * e["axes"][0]), int(0.96 * e["axes"][1])),
                e["angle_deg"], 0, 360, 255, -1)
    out[m > 0] = 200
    res = shrouded.analyze_shrouded(out, outer_diameter_mm=p.outer_diameter, eye_diameter_mm=p.eye_diameter)
    v, src, conf = T._prov(res, "blade_count")
    assert src != "measured", (v, src, conf)
    assert any("no blade pattern" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# 7 / 12 / 8 / 13: input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hint", [0, 1, -3, 49, 100, 2.5, "8", True])
def test_blade_count_hint_validation(real900, hint):
    with pytest.raises(ValueError, match="blade_count_hint"):
        shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, blade_count_hint=hint)


def test_blade_count_hint_zero_with_rim_images_no_zerodivision(real900):
    side = np.full((300, 600, 3), 90, np.uint8)
    with pytest.raises(ValueError):
        shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, blade_count_hint=0, rim_images=[side],
                                  marker_hint="bright")
    for Z in (0, 1, -2):
        with pytest.raises(ValueError, match="Z must be"):
            shrouded.analyze_rim_image(side, Z)


@pytest.mark.parametrize("h", ["left", "CW", "clockwise", "", "ccw "])
def test_invalid_handedness(real900, h):
    with pytest.raises(ValueError, match="handedness"):
        shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, handedness=h)


def test_missing_rim_image(real900, capsys):
    with pytest.raises(ValueError, match="rim_images"):
        shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, rim_images=[None])
    rc = shrouded._main([str(REAL), "--rim-image", "/nonexistent/missing.png"])
    assert rc == 2
    assert "cannot read rim image" in capsys.readouterr().err
    with pytest.raises(ValueError):
        shrouded._to_gray(None)
    with pytest.raises(ValueError):
        shrouded._to_gray(np.array([[None, None]], dtype=object))


def test_tiny_rim_image(real900):
    tiny = np.zeros((30, 40), np.uint8)
    with pytest.raises(ValueError, match="too small"):
        shrouded.analyze_rim_image(tiny, 8)
    # 60x66 used to reach find_peaks(distance=0)
    with pytest.raises(ValueError) as ei:
        shrouded.analyze_rim_image(np.zeros((60, 66), np.uint8), 8)
    assert "distance" not in str(ei.value)
    # at the minimum size the call must not die in scipy (ValueError about the gap band is fine)
    try:
        shrouded.analyze_rim_image(np.random.default_rng(0).integers(0, 255, (60, 100)).astype(np.uint8), 8)
    except ValueError as exc:
        assert "distance" not in str(exc)
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, rim_images=[tiny], handedness="cw")
    assert any("rim image 0" in w and "too small" in w for w in res.warnings), res.warnings


# ---------------------------------------------------------------------------
# 9. rim cut by the frame
# ---------------------------------------------------------------------------
def test_rim_cut_by_frame_is_flagged(real_bgr):
    H, W = real_bgr.shape[:2]
    crop = real_bgr[int(0.15 * H):int(0.85 * H), int(0.15 * W):int(0.85 * W)]   # outer rim leaves the frame
    res = shrouded.analyze_shrouded(crop, outer_diameter_mm=300.0)
    assert any("may be the inlet ring" in w for w in res.warnings), res.warnings
    for k in ("eye_diameter", "hub_diameter", "hub_bore_diameter"):
        assert T._prov(res, k)[2] <= 0.2, (k, T._prov(res, k))
    assert T._prov(res, "blade_angle_inlet")[1] == "prior"
    # the uncropped photo raises no such warning
    full = shrouded.analyze_shrouded(real_bgr, outer_diameter_mm=300.0)
    assert not any("inlet ring" in w and "may be" in w for w in full.warnings)
    assert T._prov(full, "eye_diameter")[2] >= 0.4


# ---------------------------------------------------------------------------
# 10 / 14 / 15: provenance == params after every clamp
# ---------------------------------------------------------------------------
def _assert_prov_matches_params(res):
    meta_prov = res.params.meta["provenance"]
    for k in PROV_FIELDS:
        want = getattr(res.params, k)
        assert res.provenance[k].value == pytest.approx(want, rel=1e-3, abs=0.01), \
            (k, res.provenance[k], want)
        assert meta_prov[k]["value"] == pytest.approx(want, rel=1e-3, abs=0.01), (k, meta_prov[k], want)


@pytest.mark.parametrize("kw,field", [
    ({"eye_diameter_mm": 400.0}, "eye_diameter"),         # > D2
    ({"eye_diameter_mm": 5.0}, "eye_diameter"),           # < hub
    ({"eye_diameter_mm": 20.0}, "eye_diameter"),
    ({"focal_px": 240.0}, "inner_diameter"),              # D1 >= D2 branch (camera 54 mm away)
])
def test_provenance_follows_clamps(real900, kw, field):
    s = 900.0 / 1440.0
    if "focal_px" in kw:
        kw = {"focal_px": kw["focal_px"] * s}
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=300.0, **kw)
    _assert_prov_matches_params(res)
    if field == "eye_diameter" and kw.get("eye_diameter_mm") in (400.0, 5.0, 20.0):
        assert any("clamped" in w for w in res.warnings)
        p = res.provenance["eye_diameter"]
        assert p.source == "prior" and p.confidence <= 0.1 and "clamped" in p.note


def test_outer_diameter_smaller_than_eye_clamp_consistent(real900):
    res = shrouded.analyze_shrouded(real900, outer_diameter_mm=100.0, eye_diameter_mm=180.0)
    _assert_prov_matches_params(res)
    assert res.params.eye_diameter == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 11 / 16: dtype-aware gray conversion
# ---------------------------------------------------------------------------
def test_dtype_aware_gray():
    g = cv2.imread(str(REAL), 0)
    for conv in (lambda x: x.astype(np.float32) / 255.0,
                 lambda x: x.astype(np.uint16) * 257,
                 lambda x: x.astype(np.float64)):
        out = shrouded._to_gray(conv(g))
        assert out.dtype == np.uint8
        assert np.abs(out.astype(int) - g.astype(int)).max() <= 1
    bgr = cv2.imread(str(REAL))
    out = shrouded._to_gray((bgr.astype(np.float32) / 255.0))
    assert np.abs(out.astype(int) - shrouded._to_gray(bgr).astype(int)).max() <= 1


def test_float_and_uint16_images_analyse_like_uint8(real900):
    g = cv2.cvtColor(real900, cv2.COLOR_BGR2GRAY)
    ref = shrouded.analyze_shrouded(g, outer_diameter_mm=300.0)
    for conv in (lambda x: x.astype(np.float32) / 255.0, lambda x: x.astype(np.uint16) * 257):
        res = shrouded.analyze_shrouded(conv(g), outer_diameter_mm=300.0)
        assert res.params.blade_count == ref.params.blade_count == 8
        assert res.params.eye_diameter == pytest.approx(ref.params.eye_diameter, rel=0.01)
    over = shrouded.draw_overlay(g.astype(np.float32) / 255.0, ref)
    assert over.dtype == np.uint8 and over.shape[:2] == g.shape


# ---------------------------------------------------------------------------
# misc: the provenance of a plain run is self-consistent
# ---------------------------------------------------------------------------
def test_plain_real_run_is_consistent(real_variants):
    res = real_variants["orig"]
    _assert_prov_matches_params(res)
    assert isinstance(res.params, ImpellerParams)
    res.params.validate()
    assert res.params.blade_count == 8 and T._prov(res, "blade_count")[1] == "measured"
