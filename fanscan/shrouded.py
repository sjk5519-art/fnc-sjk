"""Shrouded (production-type) impeller analyzer - fanscan v0.2, Module D.

Input: a photo *through the inlet eye* of a closed impeller (front shroud with a
conical inlet section and an inlet ring), taken under imperfect conditions
(tilt, close phone camera, cluttered desk, no reference object), optionally
plus one or more *side* photos of the rim gap with a marker on the shroud rim.
Output: :class:`ShroudedResult` - an :class:`ImpellerParams` where every field
carries a :class:`Provenance` (``measured`` / ``extrapolated`` / ``wrap`` /
``prior`` / ``user``) and a confidence in [0, 1].

Pipeline (SPEC2, Module D)
--------------------------
1. ``pose.find_ellipses`` -> rim + eye ellipses (own picker: the largest well
   supported ellipse that has a concentric family inside it; the eye is refined
   to the *inner* edge of the inlet ring).  Affine rectification on the rim.
2. Scale: reference disc > ``outer_diameter_mm`` > ``eye_diameter_mm`` > prior
   D2 = 300 mm (ratios are still measured).  With two known lengths (or a
   pinhole ``focal_px`` from EXIF) the camera distance is solved and the
   backplate-plane / eye-plane scales are perspective-corrected.
3. Eye region: the through-bore (largest dark connected component near the middle of the
   eye, hull fitted with an ellipse) anchors the impeller axis and gives the bore diameter;
   the axis is refined along the tilt direction (backplate-vs-flange parallax) within 5 % of
   the rim radius only.  Without a bore the rim centre / the Z-fold line search is used and a
   search that ends at its boundary marks the axis as uncertain.  Hub flange and bore from
   the radial profiles (circular-edge strength + angular variance).
4. Blade count: harmonic-sum FFT of the angular *deviation* profiles of the RAW polar image
   (no CLAHE: its 8x8 tile grid is centred on the axis and fakes a 4-fold pattern in a
   featureless eye), SNR-weighted vote over radii; the count needs a minimum raw grey-level
   amplitude.  The radial extent of the blade band is measured on the CLAHE image at that Z.
5. Blade root tracing: a blade seen through the eye is a *wedge* bounded by its
   root crease and its (perspective-displaced) top edge; the root crease is the
   boundary on the +theta side, i.e. the edge where, going +phi (image frame),
   one *enters* the wedge.  The entry-edge response is folded over the Z-fold
   symmetry and matched against the linear-beta camber family (beta at the inner
   and outer radius of the visible band, blade phase) - a 3-parameter matched
   filter that is immune to clutter, edge breaks and the straight leading-edge
   line.  Each blade is then traced along the template and refitted with the
   integrated camber model; traces that hug the inlet ring or the hub flange are dropped.
   Handedness comes from the sign of d(phi)/dr of the strong band edges (stable under
   rotation / scale / gamma, flips under a mirror) cross-checked with the template contrast;
   when neither is decisive it is 'undetermined' (prior flag).  beta1 is 'measured' only with
   n_good >= max(2, Z//3) agreeing, well-fitting blades, a located axis and a known hand,
   otherwise it is a prior (conf <= 0.15) with a warning; beta2 'extrapolated' with
   confidence ~ visible fraction of the blade span.
6. Wrap-angle beta2 from ``rim_images`` ('side' views of the rim gap with a
   marker on the shroud rim): gap band from the vertical-edge energy, trailing
   edges = vertical edges where the depth proxy (height of the backplate band
   visible below a surface) jumps from near to deep, marker = saturated blob at
   the rim top edge.  The marker's fractional blade index is interpolated from
   the TE positions (polynomial in the index), which needs no camera model.
   wrap mod pitch = (TE phase - LE phase) -> ``geometry.wrap_mod_pitch_candidates``
   -> ``geometry.beta2_from_wrap`` -> beta2 'wrap' (only with a perspective-corrected backplate
   scale, i.e. two known lengths or a focal length; otherwise it is reported in debug + a
   warning and the extrapolated beta2 is kept).
7. Blade widths: user > rim image (gap height / TE spacing) > prior 0.2 D2;
   b1 = user > b2 + cone prior.
8. Assembly with validation / clamping; provenance is re-synchronised with every clamped value.

Conventions: image frame y down; ``theta_math = -phi_img`` (see analyze.py);
blades sweep toward +theta_math as r grows.  In a side photo taken with the
shroud *up*, screen x increases with +theta_math (CCW seen from the inlet).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from . import geometry, pose
from .analyze import _to_gray as _analyze_to_gray
from .params import ImpellerParams
from .pose import EllipseFit

__all__ = ["Provenance", "ShroudedResult", "analyze_shrouded", "draw_overlay",
           "analyze_rim_image", "PRIOR", "SOURCES"]

SOURCES = ("measured", "extrapolated", "wrap", "prior", "user")
Z_MIN, Z_MAX = 4, 48
_COL_SCALE_FLOOR = 0.5      # per-column response scale floor as a fraction of the global scale
_EYE_MASK_SHRINK = 0.975    # eye-region mask radius as a fraction of the eye ellipse
_RIM_MIN_W, _RIM_MIN_H = 100, 60    # smallest usable side (rim-gap) image
_HAND_MF_MARGIN = 1.3       # matched-filter contrast ratio needed to call the blade hand
_HAND_OR_MIN = 0.4          # |orientation score| needed to call the blade hand from edge directions
_AXIS_SEARCH_FRAC = 0.05    # axis refinement radius around the bore centre (fraction of the rim a)
_BORE_SNAP_FRAC = 0.006     # bore centre this close to the rim centre (fraction of a) -> use the rim centre
_AXIS_MOVE_GAIN = 1.02      # template contrast gain needed to leave the bore centre
_Z_MIN_AMP = 2.0            # grey levels: minimum raw amplitude of the Z-fold pattern in the band

# Priors (ratios of a typical production-type shrouded impeller, cf. SHROUDED_PARAMS_FACTORY).
PRIOR: dict[str, float] = {
    "D2_mm": 300.0,
    "eye_over_D2": 0.60,
    "hub_over_D2": 0.30,
    "bore_over_hub": 0.0,          # unknown -> 0 (solid hub) unless seen
    "b2_over_D2": 0.20,            # SPEC2: prior blade width 0.2 D2
    "b1_over_b2": 1.30,            # SPEC2: prior b1 = 1.3 b2
    "cone_over_D2": 20.0 / 300.0,
    "ring_over_D2": 18.0 / 300.0,
    "tb_over_D2": 4.0 / 300.0,
    "ts_over_D2": 0.01,
    "t_over_D2": 0.01,             # blade thickness
    "Z": 11,
    "beta1_deg": 30.0,
    "beta2_deg": 40.0,
}


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------
@dataclass
class Provenance:
    value: float
    source: str                  # 'measured' | 'extrapolated' | 'wrap' | 'prior' | 'user'
    confidence: float            # 0..1
    note: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown provenance source {self.source!r}")
        self.value = float(self.value)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source,
                "confidence": round(self.confidence, 3), "note": self.note}


@dataclass
class ShroudedResult:
    params: ImpellerParams
    provenance: dict[str, Provenance]
    rim: EllipseFit
    eye: EllipseFit | None
    rectified: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        return list(self.params.meta.get("warnings", []))

    def provenance_table(self) -> str:
        rows = ["field                value      source        conf  note"]
        for k, p in self.provenance.items():
            rows.append(f"{k:20s} {p.value:9.3f}  {p.source:12s} {p.confidence:5.2f}  {p.note}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_gray(img: Any) -> np.ndarray:
    """Any image -> uint8 gray, rescaling by dtype (float [0, 1], uint16) exactly like the v0.1
    analyzer; raises ValueError for None / object / non-image arrays."""
    if img is None:
        raise ValueError("image is None (unreadable file?)")
    arr = np.asarray(img)
    if arr.dtype == object or arr.ndim not in (2, 3) or arr.size == 0 \
            or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise ValueError("image must be a non-empty uint8 HxW or HxWx3 array "
                         f"(got dtype {arr.dtype}, shape {arr.shape})")
    return _analyze_to_gray(arr)


def _clahe(gray: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def _polar(img: np.ndarray, centre: tuple[float, float], r_max: float, n_angle: int
           ) -> np.ndarray:
    """Polar unwrap: rows = image angle phi in [0, 2pi) (y-down frame), cols = radius
    in px (1 col = 1 px).  Returns float32 (n_angle, R)."""
    R = max(8, int(round(r_max)))
    pol = cv2.warpPolar(img, (R, int(n_angle)), (float(centre[0]), float(centre[1])), float(R),
                        cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR)
    return pol.astype(np.float32)


def _ellipse_mask(e: EllipseFit, shape: tuple[int, int], shrink: float = 1.0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    (cx, cy), (w, h), ang = e.to_cv()
    cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                (max(1, int(round(shrink * w / 2))), max(1, int(round(shrink * h / 2)))),
                ang, 0, 360, 255, -1)
    return m > 0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array as (start, end_inclusive)."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _edge_cross(prof: np.ndarray, i0: int, level: float, direction: int) -> float:
    n = len(prof)
    i = int(i0)
    above = prof[i] > level
    while 0 <= i + direction < n:
        j = i + direction
        if (prof[j] > level) != above:
            a, b = prof[i], prof[j]
            t = (level - a) / (b - a) if b != a else 0.5
            return float(i + direction * t)
        i = j
    return float(i)


def _prior_heights(D2: float, *, b2: float | None = None, b1: float | None = None,
                   cone: float | None = None, ring: float | None = None,
                   tb: float | None = None, ts: float | None = None) -> dict[str, float]:
    """Characteristic heights (mm, z up, z=0 backplate bottom) from priors / user values."""
    tb = PRIOR["tb_over_D2"] * D2 if tb is None else tb
    ts = PRIOR["ts_over_D2"] * D2 if ts is None else ts
    b2 = PRIOR["b2_over_D2"] * D2 if b2 is None else b2
    cone = PRIOR["cone_over_D2"] * D2 if cone is None else cone
    ring = PRIOR["ring_over_D2"] * D2 if ring is None else ring
    b1 = (b2 + cone if b1 is None else b1)
    return {"tb": tb, "ts": ts, "b2": b2, "b1": b1, "cone": cone, "ring": ring,
            "z_bp": tb, "z_rim": tb + b2 + ts, "z_eye": tb + b2 + cone + max(ring, ts),
            "z_hub": tb + b2 + ts}


# ---------------------------------------------------------------------------
# 1. rim / eye
# ---------------------------------------------------------------------------
def _concentric(outer: EllipseFit, inner: EllipseFit, ratio=(0.3, 0.92), centre_tol=0.15,
                aspect_tol=0.12) -> bool:
    q = inner.a / outer.a
    if not (ratio[0] <= q <= ratio[1]):
        return False
    if math.hypot(inner.center[0] - outer.center[0], inner.center[1] - outer.center[1]) > centre_tol * outer.a:
        return False
    return abs(inner.aspect - outer.aspect) <= aspect_tol


def _pick_rim_eye(ells: list[EllipseFit], shape: tuple[int, ...],
                  ) -> tuple[EllipseFit | None, EllipseFit | None, list[EllipseFit]]:
    """(rim, eye, eye_candidates).  The rim is the largest ellipse with score >= 0.4 (inside
    the image) that has a concentric smaller family (score >= 0.45) inside it; in a tilted
    view the outer silhouette is only ~half supported (backplate rim + shroud rim) and
    pose.pick_impeller_ellipses would promote the inlet ring instead."""
    if not ells:
        return None, None, []
    H, W = shape[:2]
    inside = [e for e in ells if 0 <= e.center[0] < W and 0 <= e.center[1] < H]
    best, best_fam = None, []
    for e in sorted(inside, key=lambda q: q.a, reverse=True):
        if e.score < 0.4 or e.aspect < 0.3:
            continue
        fam = [o for o in ells if o is not e and o.score >= 0.45 and _concentric(e, o)]
        if fam:
            best, best_fam = e, fam
            break
    if best is None:
        rim, eye = pose.pick_impeller_ellipses(ells, shape)
        if rim is not None and rim.aspect < 0.3:
            return None, None, []
        return rim, eye, [eye] if eye is not None else []
    # eye = best supported family member, mild preference for larger (ring outer/inner edges
    # are resolved later by _eye_inner_edge)
    eye = max(best_fam, key=lambda o: o.score + 0.15 * o.a / best.a)
    return best, eye, best_fam


def _eye_inner_edge(rect: np.ndarray, eye_r: EllipseFit, cands: list[EllipseFit],
                    debug: dict[str, Any]) -> EllipseFit:
    """Prefer the *inner* edge of the inlet ring: a concentric ellipse 1..6 % smaller in the
    candidate list, else the strongest radial edge 1..6 % inside the chosen one."""
    for c in sorted(cands, key=lambda o: o.a):
        q = c.a / eye_r.a
        if 0.93 <= q < 0.995 and c.score >= 0.4 and \
                math.hypot(c.center[0] - eye_r.center[0], c.center[1] - eye_r.center[1]) < 0.03 * eye_r.a:
            debug["eye_inner_from"] = "candidate"
            return c
    # radial profile along the normals of the eye ellipse
    t = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    P = eye_r.points(t=t)
    N = eye_r.normals(t)
    offs = np.arange(-0.08 * eye_r.a, 0.03 * eye_r.a, 1.0)
    X = P[:, None, 0] + offs[None, :] * N[:, None, 0]
    Y = P[:, None, 1] + offs[None, :] * N[:, None, 1]
    sm = cv2.GaussianBlur(rect, (0, 0), 1.0).astype(np.float32)
    samp = cv2.remap(sm, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    g = np.abs(np.gradient(samp, axis=1)).mean(axis=0)
    g = gaussian_filter1d(g, 1.0)
    pk, pr = find_peaks(g, prominence=0.15 * max(g.max(), 1e-6))
    debug["eye_edge_profile"] = {"offsets": offs, "gradient": g}
    if len(pk) == 0:
        return eye_r
    main = pk[np.argmin(np.abs(offs[pk]))]
    inner = [p for p in pk if offs[p] < offs[main] - 0.008 * eye_r.a and offs[p] > -0.065 * eye_r.a
             and g[p] >= 0.4 * g[main]]
    if inner:
        p = max(inner, key=lambda q: g[q])
        k = (eye_r.a + offs[p]) / eye_r.a
        debug["eye_inner_from"] = f"profile ({offs[p]:.1f} px)"
        return EllipseFit(eye_r.center, (eye_r.a * k, eye_r.b * k), eye_r.angle_deg,
                          eye_r.score, eye_r.n_inliers, dict(eye_r.meta))
    return eye_r


def _ring_wall_shrink(rect: np.ndarray, eye_r: EllipseFit, debug: dict[str, Any],
                      default: float = _EYE_MASK_SHRINK) -> float:
    """Mask radius (fraction of the eye ellipse) that excludes the inlet ring's inner *wall*.

    Seen through the eye at a finite distance, the cylindrical wall of the inlet ring covers
    an annulus just inside the ring edge (dark in the renders, lit in the real photo); the
    blades are hidden behind it.  Along the ellipse normals the gradient magnitude stays
    elevated (> 2x the inner baseline) over the wall band; the mask ends 1 % inside the band.
    When the eye ellipse already sits on the wall's inner edge the band is empty and
    ``default`` is returned."""
    a = eye_r.a
    t = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    P = eye_r.points(t=t)
    N = eye_r.normals(t)
    offs = np.arange(-0.15 * a, 0.0, 1.0)
    X = (P[:, None, 0] + offs[None, :] * N[:, None, 0]).astype(np.float32)
    Y = (P[:, None, 1] + offs[None, :] * N[:, None, 1]).astype(np.float32)
    sm = cv2.GaussianBlur(rect, (0, 0), 1.0).astype(np.float32)
    samp = cv2.remap(sm, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g = gaussian_filter1d(np.abs(np.gradient(samp, axis=1)).mean(axis=0), 1.5)
    debug["ring_wall_profile"] = {"offsets": offs, "gradient": g}
    inner = g[offs < -0.08 * a]
    if len(inner) < 5:
        return default
    base = float(np.median(inner))
    thr = max(2.0 * base, base + 0.1 * (float(g.max()) - base))
    # walk inward from the ring edge while the gradient stays elevated
    k = len(g) - 1
    while k > 0 and g[k] > thr:
        k -= 1
    wall_in = float(offs[k])                       # <= 0; -0.0x a when a wall band exists
    debug["ring_wall_offset_px"] = wall_in
    if wall_in > -0.012 * a:
        return default
    return float(np.clip(1.0 + (wall_in - 0.01 * a) / a, 0.9, default))


def _ellipse_rho(e: EllipseFit, pts: np.ndarray) -> np.ndarray:
    """Normalised elliptical radius of (n, 2) points w.r.t. ellipse *e* (1 = on the ellipse)."""
    c, s_ = math.cos(math.radians(e.angle_deg)), math.sin(math.radians(e.angle_deg))
    dx, dy = pts[:, 0] - e.center[0], pts[:, 1] - e.center[1]
    return np.hypot((c * dx + s_ * dy) / e.a, (-s_ * dx + c * dy) / e.b)


def _find_bore(rect: np.ndarray, eye_r: EllipseFit, c_rim: tuple[float, float],
               debug: dict[str, Any]) -> dict[str, Any] | None:
    """The hub bore: the largest dark connected component near the middle of the eye region.

    The through-bore is the one feature of the eye that is both compact, round (the rim
    rectification turns the flange plane into a near circle) and darker than everything around
    it, whatever the blades, pencil marks or cables do - so it anchors the impeller axis far
    better than the perspective-shifted rim / inlet-ring ellipse centres.  The hull of the dark
    component (cables and a lit object seen through the hole do not matter) is fitted with an
    ellipse -> centre + bore radius.  Returns None when no such blob passes the plausibility
    checks (solid hub, bore not dark, clutter): the caller then keeps the old behaviour."""
    a_eye = math.sqrt(eye_r.a * eye_r.b)
    H, W = rect.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    rr_c = np.hypot(xx - c_rim[0], yy - c_rim[1])
    in_eye = _ellipse_mask(eye_r, rect.shape, 0.9)
    disc = (rr_c < 0.6 * a_eye) & in_eye
    info: dict[str, Any] = {"found": False}
    debug["bore_search"] = info
    if disc.sum() < 500:
        info["reason"] = "search disc too small"
        return None
    sm = cv2.GaussianBlur(rect, (0, 0), max(1.0, 0.008 * a_eye))
    v = sm[disc]
    p2, med = float(np.percentile(v, 2)), float(np.median(v))
    if med - p2 < 40.0:
        info["reason"] = f"no dark region (median {med:.0f}, 2nd percentile {p2:.0f})"
        return None
    t, _ = cv2.threshold(v.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(float(t), med)
    dark = ((sm <= thr) & disc).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    order = sorted(range(1, n), key=lambda k: -st[k, cv2.CC_STAT_AREA])[:3]
    for k in order:
        area = float(st[k, cv2.CC_STAT_AREA])
        if area < 0.01 * math.pi * a_eye ** 2:
            break
        comp = lab == k
        cnts = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        pts = max(cnts, key=len)
        if len(pts) < 20:
            continue
        hull = cv2.convexHull(pts)
        (bx, by), (bw, bh), bang = cv2.fitEllipse(hull)
        ra, rb = max(bw, bh) / 2.0, min(bw, bh) / 2.0
        r_mean = 0.5 * (ra + rb)
        fill = area / (math.pi * ra * rb)
        d_far = float(rr_c[comp].max())
        if not (0.05 * a_eye <= r_mean <= 0.5 * a_eye) or rb / ra < 0.7 or fill < 0.35 \
                or d_far > 0.58 * a_eye or math.hypot(bx - c_rim[0], by - c_rim[1]) > 0.4 * a_eye:
            info["reject"] = {"r": r_mean, "aspect": rb / ra, "fill": fill}
            continue
        e = EllipseFit.from_cv(((bx, by), (bw, bh), bang))
        ring = np.zeros(rect.shape, np.uint8)
        cv2.ellipse(ring, (int(round(bx)), int(round(by))), (int(1.45 * ra), int(1.45 * rb)),
                    e.angle_deg, 0, 360, 255, -1)
        cv2.ellipse(ring, (int(round(bx)), int(round(by))), (int(1.15 * ra), int(1.15 * rb)),
                    e.angle_deg, 0, 360, 0, -1)
        ring = (ring > 0) & in_eye
        if ring.sum() < 200:
            continue
        contrast = float(np.mean(sm[ring]) - np.mean(sm[comp]))
        if contrast < 40.0:
            info["reject"] = {"contrast": contrast}
            continue
        info.update({"found": True, "thr": thr, "contrast": contrast, "fill": fill})
        return {"centre": (float(bx), float(by)), "axes": (float(ra), float(rb)),
                "angle_deg": e.angle_deg, "r_px": float(r_mean), "contrast": contrast,
                "fill": fill, "thr": thr}
    info.setdefault("reason", "no compact dark blob")
    return None


def _hand_from_orientation(img: np.ndarray, centre: tuple[float, float], mask: np.ndarray,
                           c_lo: float, c_hi: float, n_angle: int = 1440, sigma: float = 1.5
                           ) -> tuple[float, float]:
    """Sweep direction from the orientation of the strong edges in the blade band.

    In the polar unwrap (rows = image angle phi, y down; cols = radius) a blade crease is a line
    along which phi changes with r.  Edge pixels whose line is neither radial (LE lines, ribs)
    nor circular (flange / ring edges) vote with their gradient energy for the sign of
    d(phi)/dr: phi growing with r = clockwise on the screen (hand -1), shrinking = CCW (+1).
    Returns (score, energy) with score in [-1, 1]: > 0 clockwise (hand -1).  It needs no
    template fit and only depends on the axis to first order, so it is stable under rotation,
    scale and gamma, and flips under a mirror."""
    R = int(c_hi) + 3
    pol = _polar(img, centre, R, n_angle)
    mp = _polar(mask.astype(np.uint8) * 255, centre, R, n_angle) > 127
    p = gaussian_filter(pol, (sigma, sigma), mode=("wrap", "nearest"))
    gx = 0.5 * (np.roll(p, -1, 1) - np.roll(p, 1, 1))
    gy = 0.5 * (np.roll(p, -1, 0) - np.roll(p, 1, 0))
    r = np.arange(R, dtype=np.float32)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * (2.0 * math.pi / n_angle) * (-gx / gy)       # tangential px per radial px of the line
    w = gx ** 2 + gy ** 2
    sel = mp & np.isfinite(t) & (np.abs(t) > 0.2) & (np.abs(t) < 4.0) & (r >= c_lo) & (r <= c_hi)
    pos = float(w[sel & (t > 0)].sum())
    neg = float(w[sel & (t < 0)].sum())
    tot = pos + neg
    return ((pos - neg) / tot if tot > 0 else 0.0), tot


# ---------------------------------------------------------------------------
# 3. eye region: hub + bore
# ---------------------------------------------------------------------------
def _hub_bore(pol: np.ndarray, mask: np.ndarray, r_eye_px: float, blade_start: int | None,
              debug: dict[str, Any]) -> tuple[float | None, float, float | None]:
    """(r_hub_px, confidence, r_bore_px) from the polar CLAHE image, below the blade band.

    bore : outermost half-level crossing of the mean profile from the dark centre up to the
           flat hub/backplate plateau (the hole's top edge); None if the centre is not dark.
    hub  : strongest circular edge (angle-averaged |d/dr|) between the bore and the blade
           band that clearly stands out of the plateau noise; None when the hub top is shaded
           like the backplate (then the caller uses the prior).
    """
    H, R = pol.shape
    cover = mask.mean(axis=0)
    valid = cover > 0.15
    cols = np.arange(R)
    mean_prof = np.array([np.mean(pol[mask[:, c], c]) if valid[c] else np.nan for c in cols])
    std_prof = np.array([np.std(pol[mask[:, c], c]) if valid[c] else np.nan for c in cols])
    gr = np.abs(np.gradient(gaussian_filter(pol, (2.0, 1.0), mode=("wrap", "nearest")), axis=1))
    g_prof = np.array([np.mean(gr[mask[:, c], c]) if valid[c] else np.nan for c in cols])
    if (~np.isfinite(mean_prof)).all():
        return None, 0.0, None
    for a in (mean_prof, std_prof, g_prof):
        bad = ~np.isfinite(a)
        a[bad] = np.interp(cols[bad], cols[~bad], a[~bad])
    g = gaussian_filter1d(g_prof, 1.5)
    debug["profiles"] = {"mean": mean_prof, "std": std_prof, "edge": g_prof, "cover": cover}
    lo = max(3, int(0.04 * r_eye_px))
    hi = int(blade_start * 0.98) if blade_start is not None else int(0.8 * r_eye_px)
    hi = max(hi, lo + 5)
    m = gaussian_filter1d(mean_prof, 1.0)
    plateau = float(np.median(m[max(lo, int(0.55 * hi)):hi]))
    dark = float(np.median(m[lo:max(lo + 2, int(0.2 * hi))]))
    r_bore = None
    if plateau - dark > 25.0:
        level = 0.5 * (plateau + dark)
        below = np.where(m[:int(0.55 * hi)] < level)[0]
        below = below[below >= lo]
        if len(below):
            r_bore = _edge_cross(m, int(below.max()), level, +1)
            if r_bore > 0.7 * hi:
                r_bore = None
    r_hub, conf = None, 0.0
    a0 = int(max(lo, (r_bore * 1.12 + 3) if r_bore is not None else lo))
    if hi - a0 > 6:
        seg_g = g[a0:hi]
        noise = float(np.median(seg_g)) + 1e-6
        pk, pr = find_peaks(seg_g, prominence=max(0.5, 3.0 * noise))
        pk = [p + a0 for p in pk if g[p + a0] > 6.0 * noise and g[p + a0] > 1.0]
        debug["hub_edge_candidates_px"] = [int(p) for p in pk]
        if pk:
            p = max(pk, key=lambda q: g[q])
            r_hub = float(p)
            if 1 <= p < len(g) - 1:
                y0, y1, y2 = g[p - 1], g[p], g[p + 1]
                den = y0 - 2 * y1 + y2
                r_hub += float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
            conf = float(np.clip(g[p] / (noise * 12.0), 0.2, 0.9))
    else:
        debug["hub_edge_candidates_px"] = []
    if r_hub is None and blade_start is not None and hi - a0 > 6:
        # real photos: the hub flange is a bright plateau, the blade region (deeper, shadowed)
        # is darker -> the half-level crossing of the mean profile marks the flange edge
        band_level = float(np.median(m[blade_start:min(R, int(blade_start + 0.12 * r_eye_px) + 1)]))
        if plateau - band_level > 20.0:
            level = 0.5 * (plateau + band_level)
            seg = m[a0:blade_start + 1]
            above = np.where(seg > level)[0]
            if len(above):
                r_step = _edge_cross(m, a0 + int(above.max()), level, +1)
                # the blade band itself darkens the mean profile, so the step must lie inside
                # the band start; the band start is only known to ~10 px (it moves ~3 % under a
                # 20 % gamma change), hence a 4 % margin rather than a razor-edge 10 % one
                if r_step <= 0.96 * blade_start and (r_bore is None or r_step >= 1.2 * r_bore):
                    r_hub, conf = r_step, 0.35
                    debug["hub_from"] = "intensity step"
    return r_hub, conf, r_bore


# ---------------------------------------------------------------------------
# 4. blade count
# ---------------------------------------------------------------------------
def _deviation_image(pol: np.ndarray, mask: np.ndarray, c_start: int | None = None,
                     tol: float = 28.0) -> np.ndarray:
    """|pol - L(r)| with L the *backplate level* (0 outside the mask), so blade wedges of
    either polarity stand out.

    L is the per-radius median at ``c_start`` (the innermost blade radius, where the wedges
    are thin) and is then tracked outward/inward column by column as the median of the pixels
    within ``tol`` grey levels of the previous level - a plain per-column median would flip to
    the wedge level once the side-face wedges cover more than half of the circumference
    (close camera, large radius).
    """
    H, R = pol.shape
    D = np.zeros_like(pol)
    L = np.full(R, np.nan)
    ok = mask.sum(axis=0) >= 0.1 * H
    if not ok.any():
        return D
    if c_start is None or not ok[c_start]:
        c_start = int(np.argmax(ok))
    L[c_start] = float(np.median(pol[mask[:, c_start], c_start]))
    for direction in (+1, -1):
        prev = L[c_start]
        rng = range(c_start + direction, R if direction > 0 else -1, direction)
        for c in rng:
            if not ok[c]:
                L[c] = prev
                continue
            v = pol[mask[:, c], c]
            near = v[np.abs(v - prev) < tol]
            if len(near) >= max(8, 0.08 * len(v)):
                prev = float(np.median(near))
            L[c] = prev
    for c in range(R):
        if ok[c]:
            D[mask[:, c], c] = np.abs(pol[mask[:, c], c] - L[c])
    return D


def _column_spectrum(row: np.ndarray, m: np.ndarray, kmax: int, return_white: bool = False):
    """Dominant harmonic and its SNR for one polar column.

    The magnitude spectrum is whitened by a running median (13 bins) so that the smooth
    illumination hump of a real photo (large low-order harmonics) does not win, then the
    harmonic sum k, 2k, 3k is scored (a blade wedge has two edges -> strong 2k)."""
    row = np.where(m, row, np.mean(row[m]))
    row = row - row.mean()
    mag = np.abs(np.fft.rfft(row))
    n = min(len(mag) - 1, 3 * kmax + 1)
    mag = mag[:n + 1]
    bg = median_filter(mag, size=13, mode="nearest") + 1e-9
    white = mag / bg
    score = np.zeros(kmax + 1)
    for k in range(Z_MIN, kmax + 1):
        sc = white[k]
        if 2 * k <= n:
            sc += 0.7 * white[2 * k]
        if 3 * k <= n:
            sc += 0.5 * white[3 * k]
        # a strong sub-harmonic (k/2, k/3) means k is a multiple of the fundamental
        sub = 0.0
        if k % 2 == 0 and k // 2 >= Z_MIN:
            sub = max(sub, white[k // 2])
        if k % 3 == 0 and k // 3 >= Z_MIN:
            sub = max(sub, white[k // 3])
        score[k] = sc - 0.8 * sub
    k = int(np.argmax(score))
    if return_white:
        return k, float(white[k]), white
    return k, float(white[k])


def _blade_band_and_count(pol: np.ndarray, mask: np.ndarray, r_eye_px: float, r_min_px: float,
                          hint: int | None, debug: dict[str, Any], force_Z: int | None = None
                          ) -> tuple[int | None, float, float, int | None, int | None]:
    """Blade count and blade band from the Z-periodicity of the angular profiles.

    For every radius the dominant harmonic k and its SNR are computed (CLAHE polar image,
    mean removed, masked angles filled with the mean).  Z = SNR-weighted vote over the
    radii with SNR >= 2.5; the blade band = outermost run of radii whose dominant harmonic
    is Z (or 2Z: the two edges of a wedge) with SNR >= 2.5, allowing short gaps.  This is
    immune to non-periodic clutter (cables in the bore, bolt holes, pencil marks) that
    dominates an angular-variance profile in real photos.
    Returns (Z, agreement, median SNR in the band, band_start, band_end).
    """
    H, R = pol.shape
    cover = mask.mean(axis=0)
    r_max = int(min(R - 1, np.max(np.where(cover > 0.15)[0]) if (cover > 0.15).any() else R - 1))
    cols = np.arange(int(max(3, r_min_px)), r_max + 1)
    kmax = min(Z_MAX, H // 2 - 1)
    ks = np.full(R, -1)
    snr = np.zeros(R)
    whites: dict[int, np.ndarray] = {}
    for c in cols:
        m = mask[:, c]
        if m.mean() < 0.15:
            continue
        ks[c], snr[c], whites[c] = _column_spectrum(pol[:, c].astype(np.float64), m, kmax, True)
    # blades always reach the shroud lip, so the count is voted over the outer part of the
    # eye (the hub flange / bore region carries bolt holes, pencil marks and cable clutter)
    votes: dict[int, float] = {}
    for c_lo_vote in (0.45 * r_eye_px, 0.0):
        for c in cols:
            if c >= c_lo_vote and snr[c] >= 3.0 and ks[c] >= Z_MIN:
                votes[int(ks[c])] = votes.get(int(ks[c]), 0.0) + snr[c] - 2.5
        if votes:
            break
    debug["blade_count_votes"] = {int(k): round(v, 1) for k, v in votes.items()}
    debug["column_harmonic"] = ks
    debug["column_snr"] = snr
    if force_Z is not None:
        # band only: the count was decided elsewhere (on the rotation-invariant image)
        Z = int(force_Z)
        merged, tot = {Z: 1.0}, 1.0
        if Z not in range(Z_MIN, kmax + 1):
            return None, 0.0, 0.0, None, None
    else:
        if not votes:
            debug["blade_band_amp"] = 0.0
            return None, 0.0, 0.0, None, None
        # merge k and 2k votes (wedge edge pairs) in favour of the fundamental
        merged = dict(votes)
        for k, v in votes.items():
            if k % 2 == 0 and (k // 2) in votes and k // 2 >= Z_MIN:
                merged[k // 2] = merged.get(k // 2, 0.0) + 0.5 * v
        tot = sum(votes.values())
        best = max(merged.values())
        tied = sorted(k for k, v in merged.items() if v >= 0.9 * best)
        Z = hint if (hint is not None and hint in tied) else max(tied, key=lambda k: merged[k])
    # band: specific-Z test per column (single bin -> a lower threshold than the argmax vote)
    ok = np.zeros(R, bool)
    snr_z = np.zeros(R)
    for c in cols:
        w = whites.get(c)
        if w is None:
            continue
        snr_z[c] = max(float(w[Z]), float(w[2 * Z]) if 2 * Z < len(w) else 0.0)
        ok[c] = snr_z[c] >= 2.2
    debug["column_snr_z"] = snr_z
    # close short gaps (<= 3 % of the eye radius)
    gap = max(2, int(0.03 * r_eye_px))
    runs = _runs(ok)
    merged_runs: list[list[int]] = []
    for a, b in runs:
        if merged_runs and a - merged_runs[-1][1] <= gap:
            merged_runs[-1][1] = b
        else:
            merged_runs.append([a, b])
    runs = [(a, b) for a, b in merged_runs if b - a >= 0.05 * r_eye_px]
    if not runs:
        debug["blade_band_amp"] = 0.0
        return int(Z), float(merged[Z] / max(tot, 1e-9)), float(np.median(snr[cols])), None, None
    a, b = max(runs, key=lambda ab: ab[1])          # outermost run (blades reach the lip)
    band_snr = float(np.median(snr_z[a:b + 1]))
    # raw grey-level amplitude of the Z / 2Z harmonic in the band: a whitened-spectrum SNR alone
    # also fires on pure noise or a smooth gradient, the amplitude says whether there is a
    # real pattern
    amps = []
    for c in range(a, b + 1):
        m = mask[:, c]
        if m.mean() < 0.15:
            continue
        row = np.where(m, pol[:, c], float(np.mean(pol[m, c])))
        F = np.abs(np.fft.rfft(row - row.mean())) * 2.0 / len(row)
        amps.append(max(float(F[Z]), float(F[2 * Z]) if 2 * Z < len(F) else 0.0))
    debug["blade_band_amp"] = float(np.median(amps)) if amps else 0.0
    agree = float(sum(ok[a:b + 1]) / max(1, b - a + 1))
    return int(Z), agree, band_snr, int(a), int(b)


# ---------------------------------------------------------------------------
# 5. blade root tracing
# ---------------------------------------------------------------------------
def _entry_response(pol: np.ndarray, mask: np.ndarray, sigma_rows: float, c_start: int,
                    cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(W_entry, W_exit): positive part of d/dphi of the deviation image, i.e. the response of
    edges where one enters (leaves) a blade wedge going +phi (image frame).  Normalised by one
    global robust scale over the blade band *cols* (a per-column scale would turn empty
    columns into full-strength noise)."""
    D = _deviation_image(pol, mask, c_start)
    Ds = gaussian_filter(D, (sigma_rows, 1.0), mode=("wrap", "nearest"))
    G = 0.5 * (np.roll(Ds, -1, axis=0) - np.roll(Ds, 1, axis=0))
    G[~mask] = 0.0
    band = np.abs(G[:, cols])[mask[:, cols]]
    g_scale = float(np.percentile(band, 97)) + 1e-6 if band.size else 1.0
    # per-column scale with a global floor: a strong feature band (inlet-ring wall) does not
    # bury the blade edges elsewhere, and empty columns cannot amplify noise beyond 2x
    col_scale = np.array([np.percentile(np.abs(G[mask[:, c], c]), 97) if mask[:, c].sum() > 10 else g_scale
                          for c in range(G.shape[1])])
    scale = np.maximum(col_scale, _COL_SCALE_FLOOR * g_scale) + 1e-6
    G = G / scale[None, :]
    return np.maximum(G, 0.0), np.maximum(-G, 0.0)


def _camber_offsets(r_mm: np.ndarray, r_a: float, r_b: float, beta_a: np.ndarray, beta_b: np.ndarray
                    ) -> np.ndarray:
    """theta(r) [rad] of the linear-beta camber for many (beta_a, beta_b) pairs at once:
    returns (n_templates, n_r) with theta(r_a) = 0."""
    t = (r_mm - r_a) / (r_b - r_a)
    beta = np.radians(beta_a[:, None] + (beta_b - beta_a)[:, None] * t[None, :])
    dth = 1.0 / (r_mm[None, :] * np.tan(beta))
    dr = np.diff(r_mm)
    inc = 0.5 * (dth[:, 1:] + dth[:, :-1]) * dr[None, :]
    th = np.concatenate([np.zeros((len(beta_a), 1)), np.cumsum(inc, axis=1)], axis=1)
    return th


def _matched_filter(Wf: np.ndarray, r_mm: np.ndarray, dphi: float, r_a: float, r_b: float,
                    betas: np.ndarray | None = None, chunk: int = 48, hand: int = 1
                    ) -> tuple[float, float, float, float, float]:
    """Search (beta_a, beta_b, phase) maximising the folded response along the camber template.
    Wf: (per, nC) response folded over the Z blades.  Returns (beta_a, beta_b, phase_row, score,
    contrast) with the score normalised to the number of columns."""
    per, nC = Wf.shape
    if betas is None:
        b = np.arange(8.0, 86.0, 2.5)
        BA, BB = np.meshgrid(b, b, indexing="ij")
        betas = np.stack([BA.ravel(), BB.ravel()], axis=1)
    th = _camber_offsets(r_mm, r_a, r_b, betas[:, 0], betas[:, 1])          # (nT, nC)
    off = np.round(-hand * th / dphi).astype(np.int64)                      # rows (phi = -hand*theta)
    p = np.arange(per)
    best = (-np.inf, 0, 0)
    all_scores = []
    for s in range(0, len(betas), chunk):
        o = off[s:s + chunk]                                                 # (k, nC)
        idx = (p[None, :, None] + o[:, None, :]) % per                       # (k, per, nC)
        S = np.take_along_axis(Wf[None, :, :].repeat(len(o), axis=0), idx, axis=1).sum(axis=2)
        all_scores.append(S.max(axis=1))
        k, ph = np.unravel_index(int(np.argmax(S)), S.shape)
        if S[k, ph] > best[0]:
            best = (float(S[k, ph]), s + k, int(ph))
    sc = np.concatenate(all_scores)
    score, ti, ph = best
    contrast = float(score / (np.median(sc) + 1e-9))
    return float(betas[ti, 0]), float(betas[ti, 1]), float(ph), score / nC, contrast


def _fit_camber(r_mm: np.ndarray, theta: np.ndarray, r_a: float, r_b: float,
                x0: tuple[float, float] | None = None) -> tuple[float, float, float, float]:
    """Least squares of the integrated linear-beta camber -> (beta_a, beta_b, rms_deg, offset)."""
    if x0 is None:
        b1, b2 = geometry.fit_linear_beta(r_mm, theta, r_a, r_b)
        x0 = (float(np.clip(b1, 3, 177)), float(np.clip(b2, 3, 177)))

    def model(x):
        rr, th = geometry.camber_line(r_a, r_b, x[0], x[1], n=96)
        return np.interp(r_mm, rr, th) + x[2]

    off0 = float(np.median(theta - model((x0[0], x0[1], 0.0))))
    sol = least_squares(lambda x: model(x) - theta, np.array([x0[0], x0[1], off0]),
                        bounds=([2.0, 2.0, -np.inf], [178.0, 178.0, np.inf]),
                        x_scale=[10.0, 10.0, 0.1], max_nfev=300,
                        loss="soft_l1", f_scale=math.radians(0.5))
    rms = float(np.degrees(np.sqrt(np.mean(sol.fun ** 2))))
    return float(sol.x[0]), float(sol.x[1]), rms, float(sol.x[2])


def _fit_camber_joint(traces: list[tuple[np.ndarray, np.ndarray]], r_a: float, r_b: float,
                      x0: tuple[float, float]) -> tuple[float, float, float, np.ndarray]:
    """One linear-beta camber for all traces at once (per-trace angular offsets are free),
    robust loss -> (beta_a, beta_b, rms_deg, offsets)."""
    r_all = np.concatenate([t[0] for t in traces])
    th_all = np.concatenate([t[1] for t in traces])
    idx = np.concatenate([np.full(len(t[0]), i) for i, t in enumerate(traces)])
    n = len(traces)

    def resid(x):
        rr, th = geometry.camber_line(r_a, r_b, x[0], x[1], n=96)
        return np.interp(r_all, rr, th) + x[2 + idx] - th_all

    rr0, th0 = geometry.camber_line(r_a, r_b, x0[0], x0[1], n=96)
    base = np.interp(r_all, rr0, th0)
    off0 = np.array([np.median(th_all[idx == i] - base[idx == i]) for i in range(n)])
    lo = np.concatenate([[2.0, 2.0], np.full(n, -np.inf)])
    hi = np.concatenate([[178.0, 178.0], np.full(n, np.inf)])
    sol = least_squares(resid, np.concatenate([[x0[0], x0[1]], off0]), bounds=(lo, hi),
                        x_scale=np.concatenate([[10.0, 10.0], np.full(n, 0.1)]),
                        loss="soft_l1", f_scale=math.radians(0.5), max_nfev=400)
    rms = float(np.degrees(np.sqrt(np.mean(sol.fun ** 2))))
    return float(sol.x[0]), float(sol.x[1]), rms, sol.x[2:]


def _trace_blades(W: np.ndarray, mask: np.ndarray, Z: int, per: int, phase: float,
                  off: np.ndarray, cols: np.ndarray, win: int) -> list[dict[str, Any]]:
    """Follow the response ridge near the template for each blade -> per-blade traces."""
    H = W.shape[0]
    col_ref = np.percentile(W[:, cols], 90, axis=0) + 1e-6      # per column reference level
    out = []
    for i in range(Z):
        rows, vals, cc = [], [], []
        for j, c in enumerate(cols):
            pred = phase + i * per + off[j]
            w = (np.arange(-win, win + 1) + int(round(pred))) % H
            v = W[w, c]
            if not mask[int(round(pred)) % H, c]:
                continue
            k = int(np.argmax(v))
            if v[k] < 0.35 * col_ref[j] or v[k] <= 0:
                continue
            if 0 < k < len(v) - 1:
                y0, y1, y2 = v[k - 1], v[k], v[k + 1]
                den = y0 - 2 * y1 + y2
                sub = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
            else:
                sub = 0.0
            rows.append(pred + (k - win) + sub)
            vals.append(float(v[k]))
            cc.append(int(c))
        rows, vals, cc = np.asarray(rows), np.asarray(vals), np.asarray(cc)
        if len(cc) >= 5:
            # reject isolated outliers against a running median of the residual to the template
            resid = rows - (phase + i * per + np.interp(cc, cols, off))
            med = median_filter(resid, size=7, mode="nearest")
            ok = np.abs(resid - med) <= max(2.0, 0.25 * win)
            rows, vals, cc = rows[ok], vals[ok], cc[ok]
        out.append({"col": cc, "row": rows, "strength": vals})
    return out


def _support_range(tr: dict[str, Any], gap_cols: int) -> tuple[int, int] | None:
    """Longest run of traced columns without gaps larger than gap_cols -> (c_min, c_max)."""
    cc = tr["col"]
    if len(cc) < 5:
        return None
    breaks = np.where(np.diff(cc) > gap_cols)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(cc) - 1]])
    k = int(np.argmax(ends - starts))
    return int(cc[starts[k]]), int(cc[ends[k]])




def _coarse_betas(step: float) -> np.ndarray:
    b = np.arange(8.0, 86.0, step)
    BA, BB = np.meshgrid(b, b, indexing="ij")
    return np.stack([BA.ravel(), BB.ravel()], axis=1)


def _blade_analysis(rect_eq: np.ndarray, centre: tuple[float, float], eye_mask_rect: np.ndarray,
                    r_lo_px: float, r_hi_px: float, Z: int, s_bp: float, *,
                    response: str = "entry", coarse_only: bool = False, hand: int = 0,
                    debug: dict[str, Any] | None = None, eye: EllipseFit | None = None,
                    eye_shrink: float = 1.0, hub_px: float | None = None) -> dict[str, Any]:
    """Matched-filter + per-blade tracing of the root creases around *centre*.

    hand : +1 blade sweeps toward +theta_math (CCW seen from the inlet, the v0.1 convention),
           -1 the mirror image; 0 = try both and keep the better matched-filter contrast.
           theta' = -hand * phi_img is used throughout so the blade always sweeps to +theta'.
    response : 'entry' - wedge entry edges only (axial views: the root crease is always the
               +theta boundary of the side-face wedge); 'abs' - both edge polarities (tilted
               views: the parallax puts the wedge on either side, but only the root creases
               fold coherently about the backplate axis, so the matched filter still picks them).
    coarse_only : stop after the coarse matched filter (used by the centre search).
    eye, eye_shrink, hub_px : traces that spend most of their length on the inlet-ring edge
               (normalised eye radius > 0.9 * eye_shrink) or on the hub flange (radius < 1.05 *
               hub_px) are flange / ring features, not blades: they are dropped (``rejected``).

    Returns a dict with per-blade fits, the aggregate (beta_a, beta_b) on [r_a, r_b] (mm), the
    phase of the blade pattern, the per-blade phase residuals and the support range."""
    per = int(max(48, round(2 * math.pi * r_hi_px / Z)))
    per += per % 2
    H = Z * per
    dphi = 2 * math.pi / H
    R = int(math.ceil(r_hi_px)) + 2
    pol = _polar(rect_eq, centre, R, H)
    mpol = _polar(eye_mask_rect.astype(np.uint8) * 255, centre, R, H) > 127
    sigma_rows = float(np.clip(0.015 * per, 1.0, 4.0))
    c0, c1 = int(math.ceil(r_lo_px)), int(math.floor(min(r_hi_px, R - 1)))
    cols = np.arange(c0, c1 + 1)
    W_in, W_out = _entry_response(pol, mpol, sigma_rows, c0, cols)
    r_mm = cols / s_bp
    r_a, r_b = float(r_mm[0]), float(r_mm[-1])
    if response == "abs":
        W = W_in + W_out
    elif hand == -1:
        W = W_out            # mirrored hand: the root crease is the wedge *exit* edge in +phi
    else:
        W = W_in
    hands = (1, -1) if hand == 0 else (hand,)
    if coarse_only:
        sub = cols[::2]
        best = None
        results = {}
        if len(hands) == 2:
            # handedness: compare clearly non-radial templates only (the straight LE lines fold
            # perfectly for a near-radial template of either hand)
            bet = _coarse_betas(5.0)
            bet = bet[(bet[:, 0] <= 70) & (bet[:, 1] <= 70)]
        else:
            bet = _coarse_betas(5.0)
        for hd in hands:
            Wh = W if response == "abs" else (W_in if hd == 1 else W_out)
            Wf = Wh[:, sub].reshape(Z, per, len(sub)).sum(axis=0)
            ba, bb, ph, score, contrast = _matched_filter(Wf, sub / s_bp, dphi, r_a, r_b,
                                                          betas=bet, hand=hd)
            results[hd] = {"score": score, "contrast": contrast, "beta_a0": ba, "beta_b0": bb,
                           "centre": centre, "hand": hd}
        if len(hands) == 2:
            # the caller decides whether the better hand is clearly better (contrast_by_hand)
            best = dict(results[1] if results[1]["contrast"] >= results[-1]["contrast"] else results[-1])
            best["contrast_by_hand"] = {h: r["contrast"] for h, r in results.items()}
        else:
            best = results[hands[0]]
        return best
    best = None
    for hd in hands:
        Wh = W if response == "abs" else (W_in if hd == 1 else W_out)
        Wf = Wh[:, cols].reshape(Z, per, len(cols)).sum(axis=0)
        ba, bb, ph, score, contrast = _matched_filter(Wf, r_mm, dphi, r_a, r_b, hand=hd)
        if best is None or contrast > best[4]:
            best = (ba, bb, ph, score, contrast, hd, Wf, Wh)
    ba, bb, ph, score, contrast, hand, Wf, W = best
    # fine search
    fine = np.arange(-3.0, 3.01, 0.5)
    FA, FB = np.meshgrid(ba + fine, bb + fine, indexing="ij")
    betas = np.stack([FA.ravel(), FB.ravel()], axis=1)
    betas = betas[(betas[:, 0] >= 2) & (betas[:, 1] >= 2) & (betas[:, 0] <= 178) & (betas[:, 1] <= 178)]
    ba, bb, ph, score, _ = _matched_filter(Wf, r_mm, dphi, r_a, r_b, betas=betas, hand=hand)
    th = _camber_offsets(r_mm, r_a, r_b, np.array([ba]), np.array([bb]))[0]
    off = -hand * th / dphi
    win = int(max(3, round(0.10 * per)))
    traces = _trace_blades(W, mpol, Z, per, ph, off, cols, win)

    fits, phases, ranges, rejected = [], [], [], []
    gap_cols = max(3, int(0.03 * (c1 - c0)))
    for i, tr in enumerate(traces):
        rng = _support_range(tr, gap_cols)
        if rng is None:
            continue
        sel = (tr["col"] >= rng[0]) & (tr["col"] <= rng[1])
        cc, rows = tr["col"][sel], tr["row"][sel]
        if len(cc) < max(6, int(0.15 * len(cols))):
            continue
        if eye is not None or hub_px is not None:
            phi_t = rows * dphi
            on_hub = hub_px is not None and float(np.mean(cc < 1.05 * hub_px)) > 0.5
            on_ring = False
            if eye is not None:
                P = np.stack([centre[0] + cc * np.cos(phi_t), centre[1] + cc * np.sin(phi_t)], 1)
                on_ring = float(np.mean(_ellipse_rho(eye, P) > 0.9 * eye_shrink)) > 0.5
            if on_hub or on_ring:
                rejected.append((i, "hub flange" if on_hub else "inlet ring edge"))
                continue
        # the wedge apex (leading edge) and the lip occlusion distort the first/last few
        # columns (the straight LE line is traced there): trim 5 % at each end for the fit
        k = max(2, int(0.05 * len(cc)))
        cf, rf = cc[k:-k], rows[k:-k]
        theta = -hand * (rf - (ph + i * per)) * dphi         # theta' relative to the template phase
        try:
            fa, fb, rms, offs = _fit_camber(cf / s_bp, theta, r_a, r_b, x0=(ba, bb))
        except (ValueError, RuntimeError):
            continue
        fits.append({"blade": i, "beta_a": fa, "beta_b": fb, "rms_deg": rms, "n": int(len(cf)),
                     "r_min_px": float(cc[0]), "r_max_px": float(cc[-1]), "theta_offset": offs,
                     "col": cc, "row": rows})
        phases.append(offs)                                  # rad, math frame, residual vs template
        ranges.append(rng)
    out = {"per": per, "H": H, "dphi": dphi, "phase_row": ph, "beta_a0": ba, "beta_b0": bb,
           "score": score, "contrast": contrast, "r_a": r_a, "r_b": r_b, "cols": cols,
           "fits": fits, "n_traced": len(fits), "rejected": rejected, "traces": traces, "polar": pol,
           "W_in": W_in,
           "W_out": W_out, "W": W, "mask_polar": mpol, "centre": centre, "response": response,
           "hand": hand}
    if fits:
        # keep well-supported, well-fitting blades for the aggregate
        n_max = max(f["n"] for f in fits)
        good = [f for f in fits if f["n"] >= 0.5 * n_max and f["rms_deg"] <= 1.0]
        fallback = not good
        if fallback:
            good = sorted(fits, key=lambda f: f["rms_deg"] / max(f["n"], 1))[:max(2, len(fits) // 2)]
        # disagreement between ALL traced blades (a wrong trace mixed in is a red flag): IQR of beta_a
        # (badly fitting / short traces are excluded: they are noise, not disagreement)
        b_all = np.array([f["beta_a"] for f in fits if f["rms_deg"] <= 2.0 and f["n"] >= 0.4 * n_max])
        beta_iqr = float(np.percentile(b_all, 75) - np.percentile(b_all, 25)) if len(b_all) >= 3 else 0.0
        A = np.array([[f["beta_a"], f["beta_b"]] for f in good])
        beta_a, beta_b = float(np.median(A[:, 0])), float(np.median(A[:, 1]))
        spread = float(np.median(np.abs(A - np.median(A, axis=0)).max(axis=1)))
        # joint robust fit over all good traces (per-blade offsets free)
        k = max(2, int(0.05 * n_max))
        tr_list = [(f["col"][k:-k] / s_bp, -hand * (f["row"][k:-k] - (ph + f["blade"] * per)) * dphi)
                   for f in good if len(f["col"]) > 2 * k + 5]
        joint = None
        if tr_list:
            try:
                ja, jb, jrms, joff = _fit_camber_joint(tr_list, r_a, r_b, (beta_a, beta_b))
                joint = {"beta_a": ja, "beta_b": jb, "rms_deg": jrms}
            except (ValueError, RuntimeError):
                joint = None
        out.update({"beta_a": beta_a, "beta_b": beta_b, "beta_spread": spread,
                    "rms_med": float(np.median([f["rms_deg"] for f in good])),
                    "r_min_px": float(np.median([f["r_min_px"] for f in good])),
                    "r_max_px": float(np.median([f["r_max_px"] for f in good])),
                    "phase_resid": np.array(phases), "blade_ids": np.array([f["blade"] for f in fits]),
                    "n_good": 0 if fallback else len(good), "good_fallback": fallback,
                    "beta_iqr_all": beta_iqr,
                    "beta_a_agree_frac": float(np.mean(np.abs(A[:, 0] - np.median(A[:, 0])) <= 10.0)),
                    "joint": joint, "per_blade_median": (beta_a, beta_b)})
        if joint is not None and joint["rms_deg"] <= 1.5:
            out["beta_a"], out["beta_b"] = joint["beta_a"], joint["beta_b"]
    return out


def _centre_correction(ba: dict[str, Any], s_bp: float) -> tuple[float, float] | None:
    """First-order axis correction from the per-blade phase residuals: a centre error
    (dx, dy) shifts the apparent angle of blade i by (-dx sin phi_i + dy cos phi_i) / r
    (image frame) -> least squares on the residuals of the Z-fold fit."""
    if "phase_resid" not in ba or len(ba["phase_resid"]) < 4:
        return None
    per, ph, dphi = ba["per"], ba["phase_row"], ba["dphi"]
    r_ref = 0.5 * (ba["r_a"] + ba["r_b"]) * s_bp               # px
    phi_i = (ph + ba["blade_ids"] * per) * dphi                 # image-frame angle of each blade
    resid_img = -ba["hand"] * ba["phase_resid"]                 # image-frame angular residual
    A = np.stack([-np.sin(phi_i), np.cos(phi_i)], axis=1) / r_ref
    sol, *_ = np.linalg.lstsq(A, resid_img, rcond=None)
    dx, dy = float(sol[0]), float(sol[1])
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    lim = 0.12 * r_ref
    if abs(dx) >= lim or abs(dy) >= lim:
        return None          # a correction that hits its clip limit is a runaway fit, not a correction
    return float(dx), float(dy)


# ---------------------------------------------------------------------------
# marker (axial image)
# ---------------------------------------------------------------------------
def _find_marker_axial(rect: np.ndarray, centre: tuple[float, float], r_in: float, r_out: float,
                       hint: str | None) -> tuple[float, float, float] | None:
    """Saturated (bright) or very dark compact blob on the shroud face between the eye and the
    rim -> (x, y, strength) in the rectified frame."""
    H, W = rect.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - centre[0], yy - centre[1])
    ann = (rr > r_in) & (rr < r_out)
    if ann.sum() < 100:
        return None
    vals = rect[ann].astype(np.float32)
    med = float(np.median(vals))
    out = None
    modes = ["bright", "dark"] if hint not in ("bright", "dark") else [hint]
    for mode in modes:
        if mode == "bright":
            thr = max(med + 35.0, float(np.percentile(vals, 99.5)))
            m = ann & (rect >= thr)
        else:
            thr = min(med - 60.0, float(np.percentile(vals, 0.5)))
            m = ann & (rect <= thr)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        amin, amax = (0.004 * r_out) ** 2, (0.12 * r_out) ** 2
        best = None
        for k in range(1, n):
            a = stats[k, cv2.CC_STAT_AREA]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            if not (amin <= a <= amax) or a < 0.4 * w * h or max(w, h) > 3.0 * min(w, h):
                continue
            cx, cy = cent[k]
            comp = lab == k
            contrast = abs(float(rect[comp].mean()) - med)
            strength = contrast * math.sqrt(a)
            if best is None or strength > best[2]:
                best = (float(cx), float(cy), float(strength), mode)
        if best is not None and (out is None or best[2] > out[2]):
            out = best
    return out[:3] if out is not None else None


# ---------------------------------------------------------------------------
# 6. rim (side) image
# ---------------------------------------------------------------------------
def _find_marker_side(gray: np.ndarray, mode: str | None) -> tuple[float, float, tuple[int, int, int, int]] | None:
    """Saturated bright / very dark compact blob anywhere in the central 80 % of the width
    -> (x, y, bbox(x0, y0, x1, y1)) or None."""
    H, W = gray.shape
    x0, x1 = int(0.1 * W), int(0.9 * W)
    sub = gray[:, x0:x1]
    med = float(np.median(sub))
    best = None
    for m_ in (["bright", "dark"] if mode not in ("bright", "dark") else [mode]):
        mask = sub >= max(med + 50, 240) if m_ == "bright" else sub <= min(med - 80, 20)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for k in range(1, n):
            a_ = stats[k, cv2.CC_STAT_AREA]
            w_, h_ = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            if not ((0.004 * W) ** 2 <= a_ <= (0.06 * W) ** 2) or a_ < 0.5 * w_ * h_ \
                    or max(w_, h_) > 2.5 * min(w_, h_):
                continue
            strength = abs(float(sub[lab == k].mean()) - med) * math.sqrt(a_)
            if best is None or strength > best[2]:
                bx, by = stats[k, cv2.CC_STAT_LEFT] + x0, stats[k, cv2.CC_STAT_TOP]
                best = (float(cent[k][0]) + x0, float(cent[k][1]), strength, (bx, by, bx + w_, by + h_))
    return (best[0], best[1], best[3]) if best is not None else None


def analyze_rim_image(img: np.ndarray, Z: int, *, marker_hint: Any = None,
                      marker_x: float | None = None, work_width: int = 1400) -> dict[str, Any]:
    """Analyse one side view of the rim gap (shroud up, camera roughly level with the gap).

    Trailing edges are the vertical edges where the surface depth jumps from *near* (a blade
    face reaching the rim) to *deep* going +x; the depth proxy is the height of the backplate
    band (camera above the gap) or shroud-underside band (camera below) visible under/over a
    surface - a near face spans the whole gap, a deep one does not.  Occlusion silhouettes of
    the curved faces jump the other way and are rejected.

    Returns a dict with: ``gap`` (y_top, y_bot), ``te_x`` (trailing edge x, sorted),
    ``te_candidates_x`` (all vertical edges), ``marker_x``/``marker_y``, ``k_marker``
    (fractional blade index of the marker in the TE sequence), ``te_phase_deg`` =
    (phi_TE - phi_marker) mod pitch, ``spacing_px``, ``gap_px``, ``b2_over_r2`` (gap height /
    rim radius, perspective-free via the TE spacing at the image centre), ``confidence`` and
    diagnostics.  Raises ValueError when no gap band is found.
    """
    if isinstance(Z, bool) or not isinstance(Z, (int, np.integer)) or Z < 2:
        raise ValueError(f"Z must be an integer >= 2 (got {Z!r})")
    gray0 = _to_gray(img)
    if gray0.shape[0] < _RIM_MIN_H or gray0.shape[1] < _RIM_MIN_W:
        raise ValueError(f"rim image too small ({gray0.shape[1]}x{gray0.shape[0]} px; need at least "
                         f"{_RIM_MIN_W}x{_RIM_MIN_H})")
    s = min(1.0, work_width / max(gray0.shape))
    gray = cv2.resize(gray0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray0
    H, W = gray.shape
    eq = _clahe(gray)
    sm = cv2.GaussianBlur(eq, (0, 0), 1.2).astype(np.float32)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mode = marker_hint if marker_hint in ("bright", "dark") else None
    # ---- marker first (its own edges must not disturb the profiles) --------------------
    mx = my = None
    mk = None
    if marker_x is not None:
        mx = float(marker_x) * s
    else:
        mk = _find_marker_side(gray, mode)
        if mk is not None:
            mx, my = mk[0], mk[1]
    gx_c, gy_c = gx.copy(), gy.copy()
    if mk is not None:
        bx0, by0, bx1, by1 = mk[2]
        pad = max(3, int(0.004 * W))
        gx_c[max(0, by0 - pad):by1 + pad, max(0, bx0 - pad):bx1 + pad] = 0.0
        gy_c[max(0, by0 - pad):by1 + pad, max(0, bx0 - pad):bx1 + pad] = 0.0
    # ---- gap band: the two long, straight horizontal edges (shroud rim lower edge, backplate
    # top edge) that span the central columns, with vertical-edge energy (trailing edges)
    # between them.  Clutter edges are short, tilted or not paired.
    xs = slice(int(0.3 * W), int(0.7 * W))
    Ev = gaussian_filter1d(np.abs(gx_c[:, xs]).mean(axis=1), 3.0)
    Eh = gaussian_filter1d(np.abs(gy_c[:, xs]).mean(axis=1), 1.5)
    horiz = (np.abs(gy_c[:, xs]) > 2.0 * np.abs(gx_c[:, xs])) & \
            (np.abs(gy_c[:, xs]) > 0.25 * float(np.percentile(np.abs(gy_c[:, xs]), 99)))
    Lh = gaussian_filter1d(horiz.mean(axis=1), 1.5)              # fraction of columns with a horizontal edge
    pk_rows, _ = find_peaks(Lh, prominence=0.08, distance=max(1, int(0.02 * H)))
    pk_rows = [p for p in pk_rows if Lh[p] >= 0.25]
    best_pair = None
    for i, ya in enumerate(pk_rows):
        for yb in pk_rows[i + 1:]:
            hgap = yb - ya
            if not (0.06 * H <= hgap <= 0.6 * H):
                continue
            inner = Ev[int(ya + 0.15 * hgap):int(yb - 0.15 * hgap) + 1]
            if len(inner) < 3:
                continue
            ev_in = float(np.percentile(inner, 25))
            ev_out = float(np.median(np.concatenate([Ev[max(0, ya - int(0.5 * hgap)):ya], Ev[yb:min(H, yb + int(0.5 * hgap))]])) + 1e-6)
            lh_inner = float(Lh[int(ya + 0.12 * hgap):int(yb - 0.12 * hgap) + 1].max())
            sc = Lh[ya] + Lh[yb] + 0.6 * min(2.0, ev_in / ev_out) - 1.5 * lh_inner
            if best_pair is None or sc > best_pair[0]:
                best_pair = (sc, ya, yb)
    if best_pair is None:
        # fallback: vertical-edge energy band (clean background)
        lo_, hi_ = np.percentile(Ev, 20), np.percentile(Ev, 98)
        above = Ev > lo_ + 0.4 * (hi_ - lo_)
        runs = [r for r in _runs(above) if r[1] - r[0] >= 0.04 * H]
        if not runs:
            raise ValueError("rim image: no rim-gap band found (no paired horizontal edges, flat edge energy)")
        y0, y1 = max(runs, key=lambda ab: Ev[ab[0]:ab[1] + 1].sum())
        h = y1 - y0

        def nearest_edge(y_ref: int) -> int:
            win = int(max(4, 0.18 * h))
            a, b = max(0, y_ref - win), min(H - 1, y_ref + win)
            seg = Eh[a:b + 1]
            pk_, _ = find_peaks(seg, prominence=0.15 * float(seg.max()))
            pk_ = [p + a for p in pk_ if Eh[p + a] >= 0.3 * seg.max()]
            return min(pk_, key=lambda p: abs(p - y_ref)) if pk_ else a + int(np.argmax(seg))

        y_top, y_bot = nearest_edge(y0), nearest_edge(y1)
    else:
        _, y_top, y_bot = best_pair
    if y_bot - y_top < 0.03 * H:
        raise ValueError("rim image: gap band too thin")
    hg = y_bot - y_top
    # ---- vertical edges inside the gap -------------------------------------------------
    r0, r1 = int(y_top + 0.15 * hg), int(y_bot - 0.15 * hg)
    V = gaussian_filter1d(np.abs(gx_c[r0:r1 + 1]).mean(axis=0), 1.5)
    pk, pr = find_peaks(V, prominence=0.10 * float(V.max()), distance=max(1, int(0.015 * W)))
    # ---- inside-the-silhouette test: the rim's lower edge runs across every impeller column
    gy_abs = np.abs(gy_c)
    wt = int(max(3, 0.12 * hg))
    top_strength = gaussian_filter1d(gy_abs[max(0, y_top - wt):y_top + wt + 1].max(axis=0), 2.0)
    bot_strength = gaussian_filter1d(gy_abs[max(0, y_bot - wt):y_bot + wt + 1].max(axis=0), 2.0)
    ref_top = float(np.median(top_strength[xs]))
    ref_bot = float(np.median(bot_strength[xs]))
    inside = (top_strength > 0.3 * ref_top) & (bot_strength > 0.3 * ref_bot)
    # close holes narrower than ~6 % of the width (the masked marker, a cable across the rim)
    close_w = max(3, int(0.06 * W))
    inside = cv2.morphologyEx(inside.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                              np.ones((1, close_w), np.uint8))[0] > 0
    # ---- depth proxy (soft vote): band visible under (over) a surface ---------------------
    # a near face spans the whole gap; a deeper surface shows the lit backplate top below it
    # (camera above the gap) or the shroud underside above it (camera below).  The uppermost
    # strong horizontal edge in the lower 45 % of the gap (excluding the backplate's own edge)
    # gives the band height; going +x it jumps UP at a trailing edge and DOWN at an occlusion
    # silhouette.  Used only as a tie-breaker: the hard criterion is the rim-model fit below.
    thr = 0.3 * float(np.percentile(gy_abs[r0:r1 + 1], 99))
    gy_s = gaussian_filter1d(gy_abs, 2.0, axis=1)

    def band(rows: slice, from_bottom: bool) -> np.ndarray:
        blk = gy_s[rows] > thr
        out = np.zeros(W)
        for x in range(W):
            idx = np.where(blk[:, x])[0]
            if len(idx):
                out[x] = (blk.shape[0] - 1 - idx.min()) if from_bottom else (blk.shape[0] - 1 - idx.max()) * 0 + idx.max()
        return median_filter(out, size=max(3, int(0.01 * W)))

    hb = band(slice(int(y_bot - 0.45 * hg), int(y_bot - 0.06 * hg) + 1), True)
    ht = band(slice(int(y_top + 0.06 * hg), int(y_top + 0.45 * hg) + 1), False)
    d = max(2, int(0.012 * W))
    jumps = []
    ins_mask = np.zeros(len(pk), bool)
    for i, x in enumerate(pk):
        xl, xr = max(0, x - d), min(W - 1, x + d)
        ins_mask[i] = bool(inside[xl] and inside[xr])
        jb = float(np.median(hb[xr:min(W, xr + d)]) - np.median(hb[max(0, xl - d):xl + 1]))
        jt = float(np.median(ht[xr:min(W, xr + d)]) - np.median(ht[max(0, xl - d):xl + 1]))
        jumps.append((jb, jt))
    jump_sign = np.array([np.sign(jb) if abs(jb) > 0.04 * hg else (np.sign(jt) if abs(jt) > 0.04 * hg else 0.0)
                          for jb, jt in jumps])
    # ---- trailing edges = the pitch-periodic family at the rim radius ------------------------
    # rim projection: x(phi) = xc + A sin(phi) / (1 - c cos(phi)), phi = azimuth from the camera
    # direction (CCW seen from the inlet = +x on screen with the shroud up), TE_k at
    # phi0 + k * pitch.  Occlusion silhouettes of the curved faces are periodic too but sit at
    # smaller radii (compressed A) and jump the other way in depth.
    pitch = 2 * math.pi / Z
    ins_idx = np.where(ins_mask)[0]
    cand = pk[ins_idx].astype(float)
    cand_v = V[pk[ins_idx]]
    cand_js = jump_sign[ins_idx]
    sil = np.where(inside)[0]
    sil_l, sil_r = (int(sil.min()), int(sil.max())) if len(sil) else (0, W - 1)
    complete = sil_l > 0.02 * W and sil_r < 0.98 * W and (sil_r - sil_l) > 0.3 * W
    xc0 = 0.5 * (sil_l + sil_r) if complete else W / 2.0
    Rext = 0.5 * (sil_r - sil_l) if complete else 0.45 * W
    c_prior = 0.2

    def x_of_phi(phi, xc, A, c):
        return xc + A * np.sin(phi) / (1.0 - c * np.cos(phi))

    best = None
    tol = max(3.0, 0.012 * W)
    phi_grid = np.arange(0.0, pitch, math.radians(0.5))
    A_grid = np.linspace(0.9, 1.1, 11) * Rext if complete else np.linspace(0.3, 0.7, 25) * W
    ks = np.arange(-int(math.pi / 2 / pitch) - 1, int(math.pi / 2 / pitch) + 2)
    if len(cand) >= 2:
        for A in A_grid:
            for ph0 in phi_grid:
                phis = ph0 + ks * pitch
                sel = np.abs(phis) <= math.radians(78)
                xp = x_of_phi(phis[sel], xc0, A, c_prior)
                dist = np.abs(cand[:, None] - xp[None, :])
                # one candidate per prediction
                match = dist.min(axis=0) <= tol
                used = np.unique(dist.argmin(axis=0)[match])
                n_in = len(used)
                if n_in < 2:
                    continue
                res = float(np.sqrt(np.mean(dist.min(axis=0)[match] ** 2)))
                depth_vote = float(np.sum(cand_js[used] > 0) - np.sum(cand_js[used] < 0))
                score = n_in + 0.15 * depth_vote - 0.5 * res / tol + 0.02 * float(np.sum(cand_v[used])) / max(float(V.max()), 1e-6)
                if best is None or score > best[0]:
                    best = (score, n_in, used, A, ph0, res)
    te = np.array([])
    model = None
    if best is not None:
        _, n_in, used, A, ph0, res = best
        te = np.sort(cand[used])
        # refine (xc, A, phi0[, c]) on the inliers
        kk = np.round((np.arcsin(np.clip((te - xc0) / A, -1, 1)) - ph0) / pitch)

        def resid(x):
            xc_, A_, p0_, c_ = x
            return x_of_phi(p0_ + kk * pitch, xc_, A_, c_) - te

        lo_b = [xc0 - 0.1 * W, 0.5 * A, ph0 - pitch, 0.05]
        hi_b = [xc0 + 0.1 * W, 1.5 * A, ph0 + pitch, 0.5]
        x0 = [xc0, A, ph0, c_prior]
        if len(te) < 4:
            lo_b[3], hi_b[3] = c_prior - 1e-6, c_prior + 1e-6
        if len(te) < 3:
            lo_b[0], hi_b[0] = xc0 - 1e-6, xc0 + 1e-6
        try:
            sol = least_squares(resid, x0, bounds=(lo_b, hi_b), loss="soft_l1", f_scale=tol)
            xc_f, A_f, ph0_f, c_f = [float(v) for v in sol.x]
            res = float(np.sqrt(np.mean(sol.fun ** 2)))
        except ValueError:
            xc_f, A_f, ph0_f, c_f = xc0, A, ph0, c_prior
        model = {"xc": xc_f, "A": A_f, "phi0_deg": math.degrees(ph0_f), "c": c_f, "resid_px": res,
                 "k": kk.tolist()}
    # sub-pixel refine on V
    te_ref = []
    for x in te:
        x = int(round(x))
        if 0 < x < W - 1:
            y0_, y1_, y2_ = V[x - 1], V[x], V[x + 1]
            den = y0_ - 2 * y1_ + y2_
            te_ref.append(x + (float(np.clip(0.5 * (y0_ - y2_) / den, -1, 1)) if den != 0 else 0.0))
        else:
            te_ref.append(float(x))
    te = np.array(sorted(te_ref))
    depth_ok = bool(len(te) >= 2 and np.sum(cand_js[best[2]] > 0) > np.sum(cand_js[best[2]] < 0)) if best else False

    out: dict[str, Any] = {
        "scale": s, "size": (W, H), "gap": (float(y_top / s), float(y_bot / s)),
        "gap_px": float(hg / s), "te_x": (te / s).tolist(), "te_candidates_x": (pk / s).tolist(),
        "te_jumps": jumps, "depth_proxy_agrees": depth_ok, "silhouette": (sil_l / s, sil_r / s, complete),
        "marker_x": None if mx is None else float(mx / s),
        "marker_y": None if my is None else float(my / s), "n_te": int(len(te)),
        "model": model, "V": V, "hb": hb, "ht": ht, "Ev": Ev, "Eh": Eh, "inside": inside,
    }
    # ---- phase of the TE grid relative to the marker from the fitted rim model -------------
    conf = 0.0
    if mx is not None and model is not None and len(te) >= 2:
        xc_f, A_f, c_f = model["xc"], model["A"], model["c"]
        ph0_f = math.radians(model["phi0_deg"])

        def phi_of_x(xq: float) -> float | None:
            from scipy.optimize import brentq
            f = lambda p: x_of_phi(p, xc_f, A_f, c_f) - xq  # noqa: E731
            lo_p, hi_p = -math.radians(85), math.radians(85)
            if f(lo_p) * f(hi_p) > 0:
                return None
            return float(brentq(f, lo_p, hi_p))

        phi_m = phi_of_x(mx)
        if phi_m is not None:
            pitch_deg = 360.0 / Z
            te_phase = (math.degrees(ph0_f - phi_m)) % pitch_deg
            inside_span = (te.min() - 0.6 * (te[1] - te[0]) <= mx <= te.max() + 0.6 * (te[-1] - te[-2])) \
                if len(te) >= 2 else False
            spacing_c = A_f / (1.0 - c_f) * pitch                  # px per pitch at the image centre
            conf = 0.8 if (model["resid_px"] < 0.6 * tol and len(te) >= 3 and inside_span) else 0.5
            if not depth_ok:
                conf *= 0.75
            if abs(phi_m) > math.radians(45):
                conf *= 0.7
            out.update({"phi_marker_deg": math.degrees(phi_m), "te_phase_deg": float(te_phase),
                        "spacing_centre_px": float(spacing_c / s), "poly_resid_px": model["resid_px"] / s,
                        "marker_inside_span": inside_span,
                        "b2_over_r2": float(hg * pitch / spacing_c)})
    out["confidence"] = conf
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def analyze_shrouded(img: np.ndarray, *, outer_diameter_mm: float | None = None,
                     eye_diameter_mm: float | None = None, ref_diameter_mm: float | None = None,
                     blade_width_mm: float | None = None, blade_width_inlet_mm: float | None = None,
                     blade_count_hint: int | None = None,
                     rim_images: list[np.ndarray] | None = None, marker_hint: Any = None,
                     focal_px: float | None = None, hub_height_mm: float | None = None,
                     handedness: str | None = None) -> ShroudedResult:
    """Analyse a through-the-eye photo of a shrouded impeller (see module docstring).

    marker_hint : 'bright' | 'dark' | dict(axial_xy=(x, y) [original px], rim_x=[x per rim image])
    focal_px    : pinhole focal length in px (e.g. from EXIF: f35 / 36 * image width) -
                  enables the perspective correction with a single known diameter.
    handedness  : 'ccw' (blades sweep counter-clockwise with radius seen from the inlet, the
                  v0.1/CAD convention), 'cw' (mirror image) or None = detect from the image.
    """
    if handedness not in (None, "ccw", "cw"):
        raise ValueError(f"handedness must be None, 'ccw' or 'cw' (got {handedness!r})")
    if blade_count_hint is not None:
        if isinstance(blade_count_hint, (float, np.floating)) and float(blade_count_hint).is_integer():
            blade_count_hint = int(blade_count_hint)
        if isinstance(blade_count_hint, (bool, np.bool_)) or not isinstance(blade_count_hint, (int, np.integer)) \
                or not (Z_MIN <= int(blade_count_hint) <= Z_MAX):
            raise ValueError(f"blade_count_hint must be an integer in [{Z_MIN}, {Z_MAX}] "
                             f"(got {blade_count_hint!r})")
        blade_count_hint = int(blade_count_hint)
    for name_, val_ in (("outer_diameter_mm", outer_diameter_mm), ("eye_diameter_mm", eye_diameter_mm),
                        ("ref_diameter_mm", ref_diameter_mm), ("blade_width_mm", blade_width_mm),
                        ("blade_width_inlet_mm", blade_width_inlet_mm), ("focal_px", focal_px)):
        if val_ is not None and not (np.isfinite(val_) and float(val_) > 0):
            raise ValueError(f"{name_} must be a positive number (got {val_!r})")
    if rim_images is not None:
        for k_, ri_ in enumerate(rim_images):
            if ri_ is None:
                raise ValueError(f"rim_images[{k_}] is None (unreadable image file?)")
    gray = _to_gray(img)
    debug: dict[str, Any] = {}
    warnings: list[str] = []
    prov: dict[str, Provenance] = {}
    hint_dict = marker_hint if isinstance(marker_hint, dict) else {}
    hint_mode = marker_hint if marker_hint in ("bright", "dark") else hint_dict.get("mode")

    # ---- 1. ellipses + rectification ---------------------------------------
    ells, pdbg = pose.find_ellipses(gray, return_debug=True)
    rim, eye, eye_cands = _pick_rim_eye(ells, gray.shape)
    debug["ellipses"] = ells
    debug["pose_debug"] = pdbg
    if rim is None:
        raise ValueError("no rim ellipse found (is this a through-the-eye view of the impeller?)")
    if rim.aspect < 0.3:
        raise ValueError("rim ellipse is very flat: this looks like a side view, not an axial one")
    rect, M, ppu = pose.rectify(gray, rim)
    Minv = pose.invert_affine(M)
    rim_r = pose.transform_ellipse(rim, M)
    c_rim = (rect.shape[1] / 2.0, rect.shape[0] / 2.0)
    a_rim = float(rim.a)
    eye_r = pose.transform_ellipse(eye, M) if eye is not None else None
    if eye_r is not None:
        cands_r = [pose.transform_ellipse(c, M) for c in eye_cands]
        eye_r = _eye_inner_edge(rect, eye_r, cands_r, debug)
    else:
        warnings.append("eye (inlet ring) ellipse not found; eye radius taken as prior 0.6 D2 "
                        "around the rim centre")
        k = PRIOR["eye_over_D2"]
        eye_r = EllipseFit(c_rim, (a_rim * k, a_rim * k), 0.0, 0.0, 0)
    a_eye = float(math.sqrt(eye_r.a * eye_r.b))
    tilt = pose.tilt_from_ellipse(rim)
    # sanity check of the pair: with the outer rim cut by the frame the inlet ring is promoted to
    # 'rim' and the bore to 'eye' (apparent eye/rim ratio far below any production impeller)
    ratio_eye_rim = a_eye / a_rim
    rim_suspect = False
    if eye is not None and ratio_eye_rim < 0.42:
        rim_suspect = True
        warnings.append(f"picked rim ellipse may be the inlet ring (eye/rim ratio {ratio_eye_rim:.2f} < 0.42; "
                        "the outer rim leaves the frame?): diameters are unreliable")
    P_rim = rim.points(n=360)
    H_, W_ = gray.shape[:2]
    out_frac = float(np.mean((P_rim[:, 0] < 0) | (P_rim[:, 0] > W_ - 1) | (P_rim[:, 1] < 0) | (P_rim[:, 1] > H_ - 1)))
    if out_frac > 0.05:
        warnings.append(f"rim ellipse leaves the image frame ({out_frac * 100:.0f} % of its perimeter): "
                        "the outer diameter fit may be unreliable")
    debug.update({"M": M, "M_inv": Minv, "rim_rect": rim_r, "eye_rect": eye_r, "tilt_deg": tilt,
                  "px_per_unit": ppu})

    # ---- 2. scale ------------------------------------------------------------
    D2_user = outer_diameter_mm
    ref_ppm = None
    if ref_diameter_mm is not None:
        from .analyze import _find_reference_circle
        ref = _find_reference_circle(gray, debug)
        if ref is not None:
            ref_ppm = 2.0 * ref[2] / float(ref_diameter_mm)
            debug["ref_circle"] = ref
        else:
            warnings.append("reference disc not found")
    # heights model (priors unless given)
    D2_for_prior = D2_user or (eye_diameter_mm / PRIOR["eye_over_D2"] if eye_diameter_mm else None) \
        or (a_rim / ref_ppm * 2.0 if ref_ppm else None) or PRIOR["D2_mm"]
    hz = _prior_heights(D2_for_prior, b2=blade_width_mm, b1=blade_width_inlet_mm)
    if hub_height_mm is not None:
        hz["z_hub"] = float(hub_height_mm)
    cam_dist = None                                     # camera distance above the rim plane (mm)
    scale_src, scale_conf = "prior", 0.05
    if ref_ppm is not None:
        s_rim = ref_ppm                                  # disc assumed at ~the rim plane
        scale_src, scale_conf = "user", 0.8
        D2_mm = 2.0 * a_rim / s_rim
    elif D2_user is not None:
        D2_mm = float(D2_user)
        s_rim = a_rim / (0.5 * D2_mm)
        scale_src, scale_conf = "user", 0.95
    elif eye_diameter_mm is not None:
        s_rim = a_eye / (0.5 * float(eye_diameter_mm))    # far-camera assumption
        D2_mm = 2.0 * a_rim / s_rim
        scale_src, scale_conf = "user", 0.6
        warnings.append("scale from the eye ring only: far camera assumed (a close camera "
                        "enlarges the eye relative to the rim -> D2 over-estimated)")
    else:
        D2_mm = PRIOR["D2_mm"]
        s_rim = a_rim / (0.5 * D2_mm)
        warnings.append("no scale information: D2 := 300 mm prior; all lengths are relative")
    # perspective: camera distance from two lengths, or from the focal length
    if D2_user is not None and eye_diameter_mm is not None:
        cam_dist = pose.estimate_perspective_scale(rim, eye if eye is not None else eye_r,
                                                   hz["z_rim"], hz["z_eye"],
                                                   eye_over_rim_radius=eye_diameter_mm / D2_user)
        if cam_dist is None:
            warnings.append("perspective estimate ill-conditioned (eye not enlarged); far camera assumed")
    elif focal_px is not None and (D2_user is not None or eye_diameter_mm is not None):
        if D2_user is not None:
            cam_dist = float(focal_px) * 0.5 * D2_user / a_rim
        else:
            cam_dist = float(focal_px) * 0.5 * eye_diameter_mm / a_eye + (hz["z_eye"] - hz["z_rim"])
    elif scale_src != "prior":
        warnings.append("single known length: far camera assumed (no perspective correction)")
    if cam_dist is not None and cam_dist < 1.5 * (hz["z_eye"] - hz["z_rim"]):
        warnings.append(f"camera distance {cam_dist:.0f} mm implausible; ignored")
        cam_dist = None

    def scale_at(z: float) -> float:
        if cam_dist is None:
            return s_rim
        return s_rim * pose.scale_at_height(cam_dist, hz["z_rim"], z)

    s_bp, s_eye, s_hub = scale_at(hz["z_bp"]), scale_at(hz["z_eye"]), scale_at(hz["z_hub"])
    if eye_diameter_mm is not None and D2_user is None and cam_dist is not None:
        # re-derive the rim scale from the eye with the perspective now known
        s_eye = a_eye / (0.5 * eye_diameter_mm)
        s_rim = s_eye / pose.scale_at_height(cam_dist, hz["z_rim"], hz["z_eye"])
        s_bp, s_hub = scale_at(hz["z_bp"]), scale_at(hz["z_hub"])
        D2_mm = 2.0 * a_rim / s_rim
    D_eye_mm = 2.0 * a_eye / s_eye
    debug["scale"] = {"s_rim": s_rim, "s_bp": s_bp, "s_eye": s_eye, "s_hub": s_hub,
                      "cam_dist_mm": cam_dist, "heights": hz, "ratio_eye_rim_apparent": ratio_eye_rim}
    prov["outer_diameter"] = Provenance(D2_mm, scale_src if scale_src != "user" or D2_user else "measured",
                                        scale_conf if D2_user else min(scale_conf, 0.6),
                                        "rim ellipse major axis" if not D2_user else "given")
    if D2_user:
        prov["outer_diameter"] = Provenance(D2_mm, "user", 1.0, "given")
    if eye_diameter_mm is not None:
        prov["eye_diameter"] = Provenance(float(eye_diameter_mm), "user", 1.0, "given")
    else:
        prov["eye_diameter"] = Provenance(D_eye_mm, "measured" if scale_src != "prior" else "prior",
                                          (0.7 if cam_dist else 0.5) * (1.0 if eye is not None else 0.1),
                                          "inlet ring inner edge" + ("" if cam_dist else " (far camera)"))

    # ---- 3. eye region: bore (axis anchor), hub ---------------------------------------------
    rect_eq = _clahe(rect)
    # 5 % margin keeps the inlet ring's inner wall (seen under the lip) out of the eye region
    mask_shrink = _ring_wall_shrink(rect, eye_r, debug)
    debug["eye_mask_shrink"] = mask_shrink
    eye_mask = _ellipse_mask(eye_r, rect.shape, shrink=mask_shrink)
    # The impeller axis: the ellipse centres of the rim / inlet ring are perspective-shifted
    # (they sit at other heights) and a free centre search drifts onto whatever folds best
    # (inlet-ring edge, flange features).  The through-bore is a clean, round, dark blob in the
    # backplate-side flange: anchor the axis there; without a bore fall back to the rim centre.
    bore_fit = _find_bore(rect, eye_r, c_rim, debug)
    anchored = bore_fit is not None
    centre = bore_fit["centre"] if anchored else c_rim
    if anchored and math.hypot(centre[0] - c_rim[0], centre[1] - c_rim[1]) <= _BORE_SNAP_FRAC * a_rim:
        centre = c_rim          # on-axis view: the bore centre is the rim centre within the fit noise
    debug["bore_fit"] = bore_fit
    r_eye_px = a_eye
    # blade count / band / hub profiles: polar unwrap about the rim centre (the band geometry is
    # only needed to ~10 px; the axis proper is used for the blade tracing below)
    # polar radius must cover the whole eye ellipse (off-centre in tilted views)
    r_cover = float(eye_r.a * 1.02 + math.hypot(eye_r.center[0] - c_rim[0], eye_r.center[1] - c_rim[1]))
    H0 = int(np.clip(round(2 * math.pi * r_cover), 1440, 4096))
    pol0 = _polar(rect_eq, c_rim, r_cover, H0)                  # CLAHE image: hub / bore profiles
    # the blade count must not see CLAHE's 8x8 tile grid (centred on the image centre it puts a
    # spurious 4-fold pattern into a featureless eye): use the raw image - the column spectrum
    # is whitened per radius, so no further normalisation is needed and nothing depends on
    # the angle
    pol0_raw = _polar(rect, c_rim, r_cover, H0)
    mpol0 = _polar(eye_mask.astype(np.uint8) * 255, c_rim, r_cover, H0) > 127
    # Z and the evidence for it come from the raw polar image; the blade band (radial extent of
    # the Z-periodic pattern) is then measured on the CLAHE image at that Z
    Z, Z_agree, Z_snr, bs_raw, be_raw = _blade_band_and_count(
        pol0_raw, mpol0, r_eye_px, 0.08 * r_eye_px, blade_count_hint, debug)
    blade_start = blade_end = None
    if Z is not None:
        _, _, _, blade_start, blade_end = _blade_band_and_count(
            pol0, mpol0, r_eye_px, 0.08 * r_eye_px, blade_count_hint, {}, force_Z=Z)
    debug["blade_band_start_px"], debug["blade_band_end_px"] = blade_start, blade_end
    r_hub_px, hub_conf, r_bore_px = _hub_bore(pol0, mpol0, r_eye_px, blade_start, debug)
    if anchored:
        r_bore_px = bore_fit["r_px"]
    if r_hub_px is None:
        hub_mm = PRIOR["hub_over_D2"] * D2_mm
        prov["hub_diameter"] = Provenance(hub_mm, "prior", 0.1, "no hub edge found: 0.3 D2")
        warnings.append("hub flange edge not found; hub := 0.3 D2")
        r_hub_px = 0.5 * hub_mm * s_hub
    else:
        hub_mm = 2.0 * r_hub_px / s_hub
        prov["hub_diameter"] = Provenance(hub_mm, "measured" if scale_src != "prior" else "prior",
                                          hub_conf * (0.9 if hub_height_mm or cam_dist is None else 0.7),
                                          "circular edge in the eye" +
                                          ("" if hub_height_mm else "; scale at the rim plane assumed"))
    bore_mm = 2.0 * r_bore_px / s_hub if r_bore_px is not None else 0.0
    prov["hub_bore_diameter"] = Provenance(bore_mm, "measured" if r_bore_px is not None else "prior",
                                           (0.6 if anchored else 0.5) if r_bore_px is not None else 0.1,
                                           ("dark bore, hull ellipse fit" if anchored else "dark central blob")
                                           if r_bore_px is not None else "not seen")
    if rim_suspect:
        for k_ in ("eye_diameter", "hub_diameter", "hub_bore_diameter"):
            prov[k_].confidence = min(prov[k_].confidence, 0.2)
            prov[k_].note += "; rim ellipse doubtful"

    # ---- 4. blade count --------------------------------------------------------
    c_lo = int(max(r_hub_px * 1.06 + 2, blade_start if blade_start else 0, 0.08 * r_eye_px))
    c_hi = int(blade_end if blade_end else 0.97 * r_eye_px)
    blades_visible = (Z is not None and blade_start is not None and Z_snr >= 3.0
                      and Z_agree >= 0.35 and c_hi - c_lo >= 6
                      and debug.get("blade_band_amp", 0.0) >= _Z_MIN_AMP)
    if rim_suspect and not blades_visible:
        warnings.append("the picked 'rim' looks like the inlet ring and no blade pattern is visible: "
                        "every length in this result is doubtful")
    if not blades_visible:
        if blade_count_hint is not None:
            Z = int(blade_count_hint)
            prov["blade_count"] = Provenance(Z, "user", 0.9, "no blade pattern visible; hint used")
        else:
            Z = int(PRIOR["Z"])
            prov["blade_count"] = Provenance(Z, "prior", 0.05, "no blade pattern visible through the eye")
        warnings.append("no blade pattern detected through the eye (LE hidden under the shroud "
                        "lip?); blade angles are priors")
    else:
        conf = float(np.clip(0.5 * Z_agree + 0.5 * min(1.0, Z_snr / 8.0), 0.0, 1.0))
        if blade_count_hint is not None and Z != blade_count_hint:
            warnings.append(f"blade_count_hint={blade_count_hint} disagrees with detected Z={Z}")
        prov["blade_count"] = Provenance(Z, "measured", conf,
                                         f"periodicity over {c_hi - c_lo} px of radius, SNR {Z_snr:.1f}")
    debug.update({"Z": Z, "blades_visible": blades_visible, "expected_from_eye": 8,
                  "centre_rect_initial": c_rim, "r_hub_px": r_hub_px, "r_bore_px": r_bore_px,
                  "r_eye_px": r_eye_px, "axis_anchored_on_bore": anchored})

    # ---- 5. blade root tracing ---------------------------------------------------
    ba: dict[str, Any] = {}
    r1_mm = r2_mm = None
    beta1 = beta2x = None
    vis_frac = 0.0
    hand_user = {"ccw": 1, "cw": -1}.get(handedness)
    hand_final: int | None = hand_user              # +1 ccw, -1 cw, None = undetermined
    hand_src = "user" if hand_user else "none"
    axis_ok = True
    r2_mm = 0.5 * D2_mm
    if blades_visible:
        r_lo_px = max(0.1 * r_eye_px, blade_start - 0.02 * r_eye_px)
        r_hi_px = float(c_hi)
        # tilt: the eye (higher) and rim centres separate along the tilt direction
        dvec = np.array([c_rim[0] - eye_r.center[0], c_rim[1] - eye_r.center[1]])
        dlen = float(np.linalg.norm(dvec))
        tilted = dlen > 0.02 * a_rim
        response = "abs" if tilted else "entry"
        # (a prior eye circle - no inlet-ring ellipse found - is a guess, not an edge to hug)
        kw = dict(r_lo_px=r_lo_px, r_hi_px=r_hi_px, Z=Z, s_bp=s_bp, response=response,
                  eye=eye_r if eye is not None else None, eye_shrink=mask_shrink, hub_px=r_hub_px)
        band_frac0 = (r_hi_px - r_lo_px) / max(s_bp * 0.5 * D2_mm - r_lo_px, 1e-6)
        yy, xx = np.mgrid[0:rect.shape[0], 0:rect.shape[1]]
        rr_c = np.hypot(xx - centre[0], yy - centre[1])
        ann = (rr_c >= r_lo_px) & (rr_c <= r_hi_px)
        band_cover = float(eye_mask[ann].mean()) if ann.any() else 0.0
        debug["band_angular_coverage"] = band_cover
        short_band = band_frac0 < 0.25 or band_cover < 0.5
        # --- axis: the bore centre, refined by the Z-fold symmetry within a small radius ---
        mf_hand = 0 if (hand_user is None and not short_band) else (hand_user or 1)
        evals: list[tuple[tuple[float, float], dict[str, Any]]] = []

        def coarse(c: tuple[float, float]) -> dict[str, Any]:
            cs_ = _blade_analysis(rect_eq, c, eye_mask, coarse_only=True, hand=mf_hand, **kw)
            evals.append((c, cs_))
            return cs_

        search: list[tuple[float, float, tuple[float, float]]] = []
        axis_src = "bore centre" if anchored else "rim centre"
        if anchored:
            # parallax: the dark opening is seen at the flange level, the blade roots stand on the
            # backplate - the two axis points differ along the tilt direction (u).  Search that
            # line within +-_AXIS_SEARCH_FRAC * a_rim of the bore centre; the Z-fold contrast must
            # have an interior maximum, a monotonic rise only says 'further' - keep the bore centre
            L = _AXIS_SEARCH_FRAC * a_rim
            if tilted:
                u_ = dvec / dlen
                # sample the tilt line at the same stations as the free search (rim-centre based:
                # -0.4 .. 2.5 dlen), but only those within L of the bore
                t_b = (centre[0] - c_rim[0]) * u_[0] + (centre[1] - c_rim[1]) * u_[1]
                lam_all = np.linspace(-0.4 * dlen, min(2.5 * dlen, 0.2 * a_rim), 14)
                step_ = float(lam_all[1] - lam_all[0])
                lams = np.array([t for t in lam_all if abs(t - t_b) <= L] + [t_b])
                lams = np.unique(np.round(lams, 6))
                vals_ = []
                for t in lams:
                    c = (centre[0] + (t - t_b) * u_[0], centre[1] + (t - t_b) * u_[1])
                    vals_.append(coarse(c)["contrast"])
                k = int(np.argmax(vals_))
                i_b = int(np.argmin(np.abs(lams - t_b)))
                if len(lams) < 3 or k == len(lams) - 1 or k == 0:
                    # still rising at the search limit: the axis lies further away than the
                    # small neighbourhood we trust - the fold cannot confirm the bore anchor
                    axis_ok = False
                    warnings.append("Z-fold contrast still rising at the edge of the axis search around "
                                    "the bore: the impeller axis is uncertain")
                elif vals_[k] >= _AXIS_MOVE_GAIN * vals_[i_b]:
                    y0, y1, y2 = vals_[k - 1], vals_[k], vals_[k + 1]
                    den = y0 - 2 * y1 + y2
                    lam_b = float(lams[k]) + (float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) * step_
                                              if den < 0 else 0.0)
                    centre = (centre[0] + (lam_b - t_b) * u_[0], centre[1] + (lam_b - t_b) * u_[1])
                    axis_src = "bore centre + Z-fold refinement along the tilt direction"
                debug["centre_search_bore"] = [(round(float(l_ - t_b), 1), round(float(v_), 3))
                                               for l_, v_ in zip(lams, vals_)]
            else:
                coarse(centre)
        elif tilted:
            u = dvec / dlen
            # no bore: the backplate axis lies ~(z_rim - z_bp)/(z_eye - z_rim) ~ 1.7 x dlen beyond
            # the rim centre, plus the perspective shift of the ellipse centres: search up to 2.5 x dlen
            lam_max = min(2.5 * dlen, 0.2 * a_rim)
            for lam in np.linspace(-0.4 * dlen, lam_max, 14):
                c = (c_rim[0] + lam * u[0], c_rim[1] + lam * u[1])
                cs_ = coarse(c)
                search.append((float(lam), cs_["contrast"], c))
            k = int(np.argmax([s_[1] for s_ in search]))
            lam_best = search[k][0]
            if 0 < k < len(search) - 1:
                y0, y1, y2 = search[k - 1][1], search[k][1], search[k + 1][1]
                den = y0 - 2 * y1 + y2
                step = search[1][0] - search[0][0]
                if den < 0:
                    lam_best += float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) * step
            else:
                # the best fold sits at the end of the search line: not a maximum, the axis is
                # unknown (the fold is then driven by ring / flange features)
                axis_ok = False
                warnings.append("axis centre search ran to its boundary (no bore found): the "
                                "impeller axis is uncertain")
            centre = (c_rim[0] + lam_best * u[0], c_rim[1] + lam_best * u[1])
            axis_src = "Z-fold line search"
            debug["centre_search"] = [(round(s_[0], 1), round(s_[1], 3)) for s_ in search]
        else:
            coarse(centre)
        # the evaluation at the final axis (nearest evaluated centre)
        cs_final = min(evals, key=lambda e_: math.hypot(e_[0][0] - centre[0], e_[0][1] - centre[1]))[1]
        # --- handedness at the FINAL axis: edge-orientation cue + matched-filter contrast ---
        cbh = cs_final.get("contrast_by_hand")
        s_or, e_or = _hand_from_orientation(rect, centre, eye_mask, r_lo_px, r_hi_px)
        or_hand = (-1 if s_or > 0 else 1) if abs(s_or) >= _HAND_OR_MIN else None
        mf_h = None
        ratio_mf = None
        if cbh:
            cp, cm = float(cbh[1]), float(cbh[-1])
            ratio_mf = max(cp, cm) / max(min(cp, cm), 1e-9)
            if ratio_mf >= _HAND_MF_MARGIN:
                mf_h = 1 if cp > cm else -1
        debug["hand_info"] = {"orientation_score": s_or, "orientation_hand": or_hand,
                              "contrast_by_hand": cbh, "contrast_ratio": ratio_mf, "mf_hand": mf_h,
                              "short_band": short_band}
        debug["hand_contrasts"] = cbh
        if hand_user is None:
            if short_band:
                # little blade visible: only a decisive edge-orientation vote is trusted
                if abs(s_or) >= 0.6:
                    hand_final, hand_src = (-1 if s_or > 0 else 1), "orientation"
                else:
                    hand_final, hand_src = None, "prior"
            elif or_hand is not None and mf_h is not None and or_hand != mf_h:
                hand_final, hand_src = None, "conflict"
            elif or_hand is not None:
                hand_final, hand_src = or_hand, "orientation" + ("+matched filter" if mf_h else "")
            elif mf_h is not None:
                hand_final, hand_src = mf_h, "matched filter"
            else:
                hand_final, hand_src = None, "prior"
            if hand_final is None:
                warnings.append("blade sweep direction undetermined (edge orientation and template "
                                "contrast are not decisive); counter-clockwise (v0.1 convention) "
                                "assumed, blade angles are priors")
        kw["hand"] = hand_final if hand_final is not None else 1
        debug["hand"] = kw["hand"]
        ba = _blade_analysis(rect_eq, centre, eye_mask, debug=debug, **kw)
        # 2-D touch-up from the per-blade phase residuals, kept only if the fold improves; the
        # correction is refused when it hits its clip limit and may not drag an anchored axis
        # away from the bore
        corr = _centre_correction(ba, s_bp) if ba.get("n_traced", 0) >= 4 else None
        debug["centre_correction_px"] = corr
        if corr is not None and math.hypot(*corr) > 0.5:
            centre2 = (centre[0] + corr[0], centre[1] + corr[1])
            drift_ok = (not anchored) or math.hypot(centre2[0] - bore_fit["centre"][0],
                                                    centre2[1] - bore_fit["centre"][1]) \
                <= 1.1 * _AXIS_SEARCH_FRAC * a_rim
            if drift_ok:
                ba2 = _blade_analysis(rect_eq, centre2, eye_mask, debug=debug, **kw)
                if ba2.get("n_traced", 0) >= max(3, ba.get("n_traced", 0) - 1) and \
                        ba2["contrast"] >= ba["contrast"] and \
                        ba2.get("beta_spread", 99) <= ba.get("beta_spread", 99) * 1.2 + 0.5:
                    ba = ba2
                    centre = centre2
                    debug["centre_correction2_px"] = _centre_correction(ba, s_bp)
        debug["blade_analysis"] = ba
        debug["tilted_view"] = tilted
        debug["axis"] = {"source": axis_src, "anchored": anchored, "ok": axis_ok,
                         "offset_from_bore_px": (math.hypot(centre[0] - bore_fit["centre"][0],
                                                            centre[1] - bore_fit["centre"][1])
                                                 if anchored else None)}
    debug["centre_rect"] = centre
    debug["handedness"] = {"value": hand_final, "source": hand_src}
    # ---- beta1 is only 'measured' when the traces are trustworthy ------------------------
    reasons: list[str] = []
    if blades_visible:
        if "beta_a" not in ba or ba.get("n_traced", 0) == 0:
            reasons.append("no blade could be traced")
        else:
            n_need = max(2, Z // 3)
            if ba["n_good"] < n_need:
                reasons.append(f"only {ba['n_good']} of {ba['n_traced']} traced blades fit well "
                               f"(need {n_need} of {Z})")
            if ba.get("beta_iqr_all", 0.0) > 15.0:
                reasons.append(f"traced blades disagree (beta IQR {ba['beta_iqr_all']:.0f} deg)")
            elif ba.get("beta_a_agree_frac", 1.0) < 0.7:
                reasons.append(f"the well-fitting blades disagree on beta1 (only "
                               f"{ba['beta_a_agree_frac'] * 100:.0f} % within 10 deg of the median)")
        if hand_final is None:
            reasons.append("sweep direction undetermined")
        if not axis_ok:
            reasons.append("axis not located")
        if ba.get("rejected"):
            debug["traces_rejected"] = ba["rejected"]
    if blades_visible and not reasons:
        r_a, r_b = ba["r_a"], ba["r_b"]
        beta_a, beta_b = ba["beta_a"], ba["beta_b"]
        slope = (beta_b - beta_a) / (r_b - r_a)
        r_min_mm = ba["r_min_px"] / s_bp
        r_max_mm = ba["r_max_px"] / s_bp
        le_seen = r_min_mm > (r_a + 0.04 * (r_b - r_a))
        if le_seen:
            r1_mm = r_min_mm
            prov["inner_diameter"] = Provenance(2 * r1_mm, "measured" if scale_src != "prior" else "prior",
                                                0.6, "wedge apex = blade leading edge root")
        else:
            r1_mm = r_min_mm
            prov["inner_diameter"] = Provenance(2 * r1_mm, "prior", 0.2,
                                                "LE not seen (trace starts at the hub/search limit)")
            warnings.append("blade leading edge not seen: D1 := innermost traced radius")
        beta1 = beta_a + slope * (r1_mm - r_a)
        vis_frac = float(np.clip((r_max_mm - r1_mm) / max(r2_mm - r1_mm, 1e-6), 0.0, 1.0))
        fitq = float(np.clip(1.0 - ba["beta_spread"] / 10.0, 0.1, 1.0)
                     * np.clip(ba["n_good"] / Z, 0.3, 1.0)
                     * np.clip(1.0 - ba["rms_med"] / 3.0, 0.3, 1.0))
        # the slope d(beta)/dr is poorly determined from a short band or inconsistent blades:
        # shrink it toward the prior slope (beta2 prior) with a weight ~ band length x consistency
        band_frac = (r_b - r_a) / max(r2_mm - r1_mm, 1e-6)
        w_slope = float(np.clip(band_frac / 0.35, 0.0, 1.0) * np.clip(1.0 - ba["beta_spread"] / 8.0, 0.0, 1.0)
                        * min(1.0, ba["n_good"] / 3.0))
        slope_prior = (PRIOR["beta2_deg"] - beta1) / max(r2_mm - r1_mm, 1e-6)
        slope_used = w_slope * slope + (1.0 - w_slope) * slope_prior
        beta2x = beta_a + slope_used * (r2_mm - r_a)
        debug["slope"] = {"fit": slope, "prior": slope_prior, "weight": w_slope, "band_frac": band_frac}
        prov["blade_angle_inlet"] = Provenance(beta1, "measured", 0.9 * fitq,
                                               f"{ba['n_good']}/{Z} blades, spread {ba['beta_spread']:.1f} deg")
        if hand_final == -1:
            warnings.append("blades sweep clockwise with radius (seen from the inlet): mirror image of "
                            "the v0.1/CAD convention; angles are reported for the mirrored blade")
        prov["blade_angle_outlet"] = Provenance(beta2x, "extrapolated", vis_frac * fitq * (0.5 + 0.5 * w_slope),
                                                f"linear-beta extrapolation, visible {vis_frac:.2f} of span"
                                                + (f", slope shrunk toward prior (w={w_slope:.2f})" if w_slope < 0.8 else ""))
    else:
        if blades_visible:
            warnings.append("blade angles are priors, not measured: " + "; ".join(reasons))
            if "beta_a" in ba:
                debug["beta1_rejected"] = {"beta_a": ba["beta_a"], "beta_b": ba["beta_b"],
                                           "n_good": ba["n_good"], "n_traced": ba["n_traced"],
                                           "reasons": reasons}
            if hand_final == -1:
                warnings.append("blades appear to sweep clockwise (seen from the inlet): mirror image of "
                                "the v0.1/CAD convention")
        r1_mm = PRIOR["eye_over_D2"] * r2_mm
        prov["inner_diameter"] = Provenance(2 * r1_mm, "prior", 0.1, "LE hidden: D1 := eye diameter prior")
        beta1, beta2x = PRIOR["beta1_deg"], PRIOR["beta2_deg"]
        prov["blade_angle_inlet"] = Provenance(beta1, "prior", 0.05, "typical backward-curved")
        prov["blade_angle_outlet"] = Provenance(beta2x, "prior", 0.05, "typical backward-curved")
    debug["visible_fraction"] = vis_frac

    # ---- marker in the axial image -------------------------------------------------
    marker_axial = None
    if "axial_xy" in hint_dict:
        p = pose.apply_affine(M, np.array([hint_dict["axial_xy"]], float))[0]
        marker_axial = (float(p[0]), float(p[1]), float("inf"))
    elif rim_images:
        marker_axial = _find_marker_axial(rect, centre, 1.04 * r_eye_px, 1.03 * a_rim, hint_mode)
    theta_marker = None
    if marker_axial is not None:
        theta_marker = math.degrees(-math.atan2(marker_axial[1] - centre[1], marker_axial[0] - centre[0]))
    debug["marker_axial_rect"] = marker_axial
    debug["marker_theta_deg"] = theta_marker

    # ---- 6. wrap angle from rim images ------------------------------------------------
    beta2 = beta2x
    rim_results: list[dict[str, Any]] = []
    b2_side: list[float] = []
    if rim_images:
        rim_xs = hint_dict.get("rim_x")
        for k, rimg in enumerate(rim_images):
            try:
                rr = analyze_rim_image(rimg, Z, marker_hint=hint_mode,
                                       marker_x=(rim_xs[k] if rim_xs and k < len(rim_xs) else None))
            except ValueError as exc:
                warnings.append(f"rim image {k}: {exc}")
                continue
            rim_results.append(rr)
            if rr.get("b2_over_r2"):
                b2_side.append(rr["b2_over_r2"] * r2_mm)
        debug["rim_images"] = rim_results
        good = [rr for rr in rim_results if "te_phase_deg" in rr]
        if theta_marker is None:
            warnings.append("marker not found in the axial image; wrap-angle beta2 skipped")
        elif not good:
            warnings.append("no rim image yielded a marker + trailing edges; wrap-angle beta2 skipped")
        elif prov["blade_angle_inlet"].source != "measured" or "fits" not in ba or not ba["fits"]:
            warnings.append("beta1 not measured from the axial image; wrap-angle beta2 skipped")
        else:
            pitch = 360.0 / Z
            # LE phase relative to the marker: theta of each blade at r1 (template phase + fit offset)
            ph_row, per, dphi = ba["phase_row"], ba["per"], ba["dphi"]
            r_a = ba["r_a"]
            # the per-blade theta_offset is the blade angle at r_a (the camber model starts
            # there), the wrap model integrates from r1: carry the phase from r_a to r1 along
            # the fitted camber (r1 = innermost traced radius >= r_a, so the sweep is positive)
            th_a_to_r1 = 0.0
            if r1_mm > r_a + 1e-9:
                _, th = geometry.camber_line(r_a, r1_mm, ba["beta_a"], beta1, n=64)
                th_a_to_r1 = float(th[-1])
            elif r1_mm < r_a - 1e-9:
                _, th = geometry.camber_line(r1_mm, r_a, beta1, ba["beta_a"], n=64)
                th_a_to_r1 = -float(th[-1])
            hand = ba["hand"]
            le_phases = []
            for f in ba["fits"]:
                # theta' of the blade root at r1 (theta' = -hand*phi, blade sweeps to +theta')
                th_le = -hand * (ph_row + f["blade"] * per) * dphi + f["theta_offset"] + th_a_to_r1
                le_phases.append((math.degrees(th_le) - hand * theta_marker) % pitch)
            le_ph = np.array(le_phases)
            # circular median modulo pitch
            ang = le_ph / pitch * 2 * math.pi
            le_phase = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 360.0 * pitch) % pitch
            # side view: screen x grows with the CCW azimuth (= +theta for hand +1); for the
            # mirrored hand theta' = -theta so the TE phase flips sign
            te_phases = np.array([(hand * rr["te_phase_deg"]) % pitch for rr in good])
            w = np.array([rr["confidence"] for rr in good])
            ang = te_phases / pitch * 2 * math.pi
            te_phase = (math.degrees(math.atan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) / 360.0 * pitch) % pitch
            te_scatter = float(np.degrees(np.sqrt(-2 * np.log(max(1e-9, np.hypot(
                (w * np.sin(ang)).sum() / w.sum(), (w * np.cos(ang)).sum() / w.sum())))))) * pitch / 360.0 \
                if len(good) > 1 else 0.0
            wrap_mod = (te_phase - le_phase) % pitch
            est_wrap = geometry.wrap_from_betas(r1_mm, r2_mm, beta1, float(np.clip(beta2x, 5, 170)))
            cands = geometry.wrap_mod_pitch_candidates(wrap_mod, Z, est_wrap)
            chosen = None
            for wv in cands:
                try:
                    chosen = (wv, geometry.beta2_from_wrap(r1_mm, r2_mm, beta1, wv))
                    break
                except ValueError:
                    continue
            debug["wrap"] = {"le_phase_deg": le_phase, "te_phase_deg": te_phase, "wrap_mod_deg": wrap_mod,
                             "est_wrap_deg": est_wrap, "candidates": cands, "te_scatter_deg": te_scatter,
                             "le_phases": le_ph.tolist(), "te_phases": te_phases.tolist()}
            if chosen is None:
                warnings.append("wrap angle not attainable with the measured beta1; keeping extrapolated beta2")
            else:
                wrap_deg, beta2_w = chosen
                amb = abs(cands[0] - est_wrap)
                conf = float(np.mean(w)) * (1.0 if amb < pitch / 3 else 0.6) * (1.0 if te_scatter < 3 else 0.6)
                debug["wrap"].update({"wrap_deg": wrap_deg, "beta2_wrap": beta2_w})
                if cam_dist is None:
                    # the wrap integral depends on r1 in the BACKPLATE scale (ln(r2/r1)/tan beta):
                    # with a single known length the close-camera enlargement is uncorrected, a 17 %
                    # error in r1 moves beta2 by ~15 deg - worse than the extrapolated value
                    debug["wrap"]["rejected"] = "no perspective correction (single known length)"
                    warnings.append(f"wrap-angle beta2 = {beta2_w:.1f} deg (conf {conf:.2f}) not used: only one "
                                    "known length, backplate scale uncorrected; keeping extrapolated beta2 "
                                    f"{beta2x:.1f} deg (give outer_diameter_mm AND eye_diameter_mm, or focal_px)")
                else:
                    beta2 = beta2_w
                    prov["blade_angle_outlet"] = Provenance(beta2, "wrap", conf,
                                                            f"wrap {wrap_deg:.1f} deg from {len(good)} rim image(s)"
                                                            f" (extrapolated {beta2x:.1f} deg)")

    # ---- 7. blade widths ----------------------------------------------------------------
    if blade_width_mm is not None:
        b2 = float(blade_width_mm)
        prov["blade_width"] = Provenance(b2, "user", 1.0, "given")
    elif b2_side:
        b2 = float(np.median(b2_side))
        prov["blade_width"] = Provenance(b2, "measured", 0.6, "rim gap height / TE spacing in the side view")
    else:
        b2 = PRIOR["b2_over_D2"] * D2_mm
        prov["blade_width"] = Provenance(b2, "prior", 0.1, "0.2 D2 (not observable axially)")
        warnings.append("blade width b2 not observable in an axial photo; 0.2 D2 assumed")
    if blade_width_inlet_mm is not None:
        b1 = float(blade_width_inlet_mm)
        prov["blade_width_inlet"] = Provenance(b1, "user", 1.0, "given")
    else:
        b1 = PRIOR["b1_over_b2"] * b2
        prov["blade_width_inlet"] = Provenance(b1, "prior", 0.1, "1.3 b2")
    cone = max(b1 - b2, 0.0)
    ring = PRIOR["ring_over_D2"] * D2_mm
    prov["shroud_cone_height"] = Provenance(cone, "prior", 0.1, "b1 - b2")
    prov["shroud_inlet_height"] = Provenance(ring, "prior", 0.1, "0.06 D2")

    # ---- 8. assemble ---------------------------------------------------------------------
    t_mm = PRIOR["t_over_D2"] * D2_mm
    prov["blade_thickness"] = Provenance(t_mm, "prior", 0.1, "0.01 D2")
    D1 = 2.0 * r1_mm
    D_eye = prov["eye_diameter"].value
    hub = prov["hub_diameter"].value
    # clamp into a valid ordering (provenance is re-synchronised with every clamped value below)
    if not hub < D1:
        warnings.append(f"hub {hub:.1f} >= D1 {D1:.1f}: hub clamped to 0.9 D1")
        hub = 0.9 * D1
    if not D1 < D2_mm:
        warnings.append("D1 >= D2: D1 clamped")
        D1 = 0.8 * D2_mm
    if not (hub < D_eye <= D2_mm):
        warnings.append("eye diameter outside (hub, D2]: clamped")
        D_eye = float(np.clip(D_eye, hub * 1.05, D2_mm))
    bore = prov["hub_bore_diameter"].value
    if bore >= hub:
        warnings.append(f"bore {bore:.1f} >= hub {hub:.1f}: bore dropped")
        bore = 0.0
    b1_ang = float(np.clip(beta1, 1.0, 179.0))
    b2_ang = float(np.clip(beta2, 1.0, 179.0))
    t_mm = min(t_mm, 0.24 * (0.5 * D2_mm - 0.5 * D1))
    final_vals = {"outer_diameter": D2_mm, "inner_diameter": D1, "hub_diameter": hub,
                  "eye_diameter": D_eye, "hub_bore_diameter": bore, "blade_angle_inlet": b1_ang,
                  "blade_angle_outlet": b2_ang, "blade_thickness": t_mm, "blade_width": b2,
                  "blade_width_inlet": b1, "shroud_cone_height": cone, "shroud_inlet_height": ring}
    for k_, v_ in final_vals.items():
        old_ = prov[k_]
        if abs(old_.value - v_) > 1e-6 * max(1.0, abs(v_)):
            prov[k_] = Provenance(v_, "prior", min(old_.confidence, 0.1),
                                  f"clamped from {old_.value:.4g} ({old_.source}) to a valid range")
    params = ImpellerParams(
        outer_diameter=round(D2_mm, 3), inner_diameter=round(D1, 3), hub_diameter=round(hub, 3),
        blade_count=int(Z), blade_angle_inlet=round(b1_ang, 2), blade_angle_outlet=round(b2_ang, 2),
        blade_width=round(b2, 3), blade_thickness=round(t_mm, 3),
        backplate_thickness=round(hz["tb"], 3), hub_height=(round(hub_height_mm, 3) if hub_height_mm else None),
        shroud=True, shroud_thickness=round(hz["ts"], 3), eye_diameter=round(D_eye, 3),
        shroud_cone_height=round(cone, 3), shroud_inlet_height=round(ring, 3),
        hub_bore_diameter=round(bore, 3), blade_width_inlet=round(b1, 3),
        meta={"source": "fanscan.shrouded", "warnings": warnings,
              "px_per_mm_rim": float(s_rim), "px_per_mm_backplate": float(s_bp),
              "camera_distance_mm": cam_dist, "tilt_deg": float(tilt),
              "handedness": {1: "ccw", -1: "cw"}.get(hand_final, "undetermined"),
              "handedness_source": hand_src,
              "visible_blade_fraction": float(vis_frac),
              "provenance": {k: v.to_dict() for k, v in prov.items()},
              "per_blade_beta": [[round(f["beta_a"], 2), round(f["beta_b"], 2)] for f in ba.get("fits", [])],
              "blade_count_votes": {str(k): v for k, v in debug.get("blade_count_votes", {}).items()}},
    )
    try:
        params.validate()
    except ValueError as exc:
        warnings.append(f"params.validate(): {exc}")
    debug.update({"rectified_shape": rect.shape, "warnings": warnings})
    return ShroudedResult(params=params, provenance=prov, rim=rim, eye=eye, rectified=rect, debug=debug)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def draw_overlay(img: np.ndarray, result: ShroudedResult) -> np.ndarray:
    """BGR copy of the ORIGINAL image with rim/eye ellipses, hub circle, blade root traces,
    the extrapolated camber to r2 (dashed), the marker and a text block."""
    gray = _to_gray(np.asarray(img))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    d = result.debug
    Minv = d["M_inv"]
    p = result.params
    thick = max(1, int(round(max(gray.shape) / 700)))

    def to_orig(pts):
        return pose.apply_affine(Minv, np.asarray(pts, float).reshape(-1, 2))

    def poly(pts_rect, colour, dashed=False, t=thick):
        P = to_orig(pts_rect)
        if dashed:
            for i in range(0, len(P) - 1, 2):
                cv2.line(out, tuple(np.round(P[i]).astype(int)), tuple(np.round(P[i + 1]).astype(int)),
                         colour, t, cv2.LINE_AA)
        else:
            cv2.polylines(out, [np.round(P).astype(np.int32).reshape(-1, 1, 2)], False, colour, t, cv2.LINE_AA)

    for e, col in ((result.rim, (0, 220, 0)), (result.eye, (0, 0, 255))):
        if e is None:
            continue
        (cx, cy), (w, h), ang = e.to_cv()
        cv2.ellipse(out, (int(round(cx)), int(round(cy))), (int(round(w / 2)), int(round(h / 2))),
                    ang, 0, 360, col, thick, cv2.LINE_AA)
    if d.get("eye_rect") is not None and result.eye is None:
        ee = d["eye_rect"]
        poly(ee.points(n=180), (0, 0, 255))
    centre = d.get("centre_rect", d.get("centre_rect_initial"))
    if centre is not None:
        t = np.linspace(0, 2 * math.pi, 181)
        for rpx, col in ((d.get("r_hub_px"), (255, 160, 0)), (d.get("r_bore_px"), (255, 0, 200))):
            if rpx:
                poly(np.stack([centre[0] + rpx * np.cos(t), centre[1] + rpx * np.sin(t)], 1), col)
        c0 = to_orig([centre])[0]
        cv2.drawMarker(out, tuple(np.round(c0).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 20, thick)
    ba = d.get("blade_analysis", {})
    s_bp = p.meta.get("px_per_mm_backplate", 1.0)
    if ba.get("fits"):
        ph, per, dphi = ba["phase_row"], ba["per"], ba["dphi"]
        r_a, r_b, r2 = ba["r_a"], ba["r_b"], p.r2
        beta_a, beta_b = ba.get("beta_a", ba["beta_a0"]), ba.get("beta_b", ba["beta_b0"])
        beta_r2 = float(np.clip(beta_a + (beta_b - beta_a) * (r2 - r_a) / (r_b - r_a), 2, 178))
        rr, th = geometry.camber_line(r_a, r2, beta_a, beta_r2, n=80)
        rr_b, th_b = geometry.camber_line(r_a, r_b, beta_a, beta_b, n=96)
        for f in ba["fits"]:
            phi = f["row"] * dphi
            pts = np.stack([centre[0] + f["col"] * np.cos(phi), centre[1] + f["col"] * np.sin(phi)], 1)
            poly(pts, (0, 255, 255), t=thick + 1)
            # aggregate camber (joint fit) through this blade's trace, extrapolated to r2 (dashed)
            k = max(2, int(0.05 * len(f["col"])))
            hand = ba.get("hand", 1)
            th_tr = -hand * (f["row"][k:-k] - (ph + f["blade"] * per)) * dphi
            off = float(np.median(th_tr - np.interp(f["col"][k:-k] / s_bp, rr_b, th_b)))
            phi_e = (ph + f["blade"] * per) * dphi - hand * (th + off)
            pts = np.stack([centre[0] + rr * s_bp * np.cos(phi_e), centre[1] + rr * s_bp * np.sin(phi_e)], 1)
            poly(pts, (255, 200, 0), dashed=True)
    mk = d.get("marker_axial_rect")
    if mk is not None:
        m0 = to_orig([mk[:2]])[0]
        cv2.circle(out, tuple(np.round(m0).astype(int)), 14, (255, 0, 255), thick + 1, cv2.LINE_AA)
    pv = result.provenance
    lines = [f"Z={p.blade_count} ({pv['blade_count'].source} {pv['blade_count'].confidence:.2f})",
             f"beta1={p.blade_angle_inlet:.1f} ({pv['blade_angle_inlet'].source} {pv['blade_angle_inlet'].confidence:.2f})",
             f"beta2={p.blade_angle_outlet:.1f} ({pv['blade_angle_outlet'].source} {pv['blade_angle_outlet'].confidence:.2f})",
             f"D2={p.outer_diameter:.0f} ({pv['outer_diameter'].source})  Deye={p.eye_diameter:.0f} ({pv['eye_diameter'].source})",
             f"D1={p.inner_diameter:.0f} ({pv['inner_diameter'].source})  hub={p.hub_diameter:.0f} ({pv['hub_diameter'].source})",
             f"tilt {p.meta.get('tilt_deg', 0):.1f} deg, visible {p.meta.get('visible_blade_fraction', 0):.2f}"]
    fs = max(0.5, gray.shape[1] / 1600)
    y = int(28 * fs) + 6
    for ln in lines:
        cv2.putText(out, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)
        y += int(30 * fs)
    return out


def draw_rim_overlay(img: np.ndarray, rr: dict[str, Any]) -> np.ndarray:
    """BGR copy of a rim image with the gap band, TE positions and the marker drawn."""
    gray = _to_gray(np.asarray(img))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    W = out.shape[1]
    yt, yb = rr["gap"]
    for y, col in ((yt, (0, 220, 0)), (yb, (0, 220, 0))):
        cv2.line(out, (0, int(y)), (W - 1, int(y)), col, 2)
    for x in rr.get("te_candidates_x", []):
        cv2.line(out, (int(x), int(yt)), (int(x), int(yb)), (120, 120, 120), 1)
    for x in rr["te_x"]:
        cv2.line(out, (int(x), int(yt)), (int(x), int(yb)), (0, 0, 255), 2)
    if rr.get("marker_x") is not None:
        cv2.circle(out, (int(rr["marker_x"]), int(rr.get("marker_y") or yt)), 12, (255, 0, 255), 2)
    if "te_phase_deg" in rr:
        cv2.putText(out, f"TE phase {rr['te_phase_deg']:.1f} deg, marker phi={rr['phi_marker_deg']:.1f}, conf {rr['confidence']:.2f}",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Analyse a through-the-eye photo of a shrouded impeller")
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="params JSON")
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--outer-diameter-mm", type=float, default=None)
    ap.add_argument("--eye-diameter-mm", type=float, default=None)
    ap.add_argument("--blade-width-mm", type=float, default=None)
    ap.add_argument("--blade-count-hint", type=int, default=None)
    ap.add_argument("--rim-image", action="append", default=[])
    ap.add_argument("--marker", choices=["bright", "dark"], default=None)
    ap.add_argument("--focal-px", type=float, default=None)
    a = ap.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        print("cannot read", a.image, file=sys.stderr)
        return 2
    rims = [cv2.imread(pth) for pth in a.rim_image]
    for pth, rim_img in zip(a.rim_image, rims):
        if rim_img is None:
            print("cannot read rim image", pth, file=sys.stderr)
            return 2
    res = analyze_shrouded(img, outer_diameter_mm=a.outer_diameter_mm, eye_diameter_mm=a.eye_diameter_mm,
                           blade_width_mm=a.blade_width_mm, blade_count_hint=a.blade_count_hint,
                           rim_images=rims or None, marker_hint=a.marker, focal_px=a.focal_px)
    print(res.provenance_table())
    for w in res.warnings:
        print("warning:", w)
    if a.out:
        res.params.save(a.out)
    if a.overlay:
        cv2.imwrite(a.overlay, draw_overlay(img, res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
