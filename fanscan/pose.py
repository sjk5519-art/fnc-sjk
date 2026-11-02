"""Rim / eye ellipse detection and first-order rectification (fanscan v0.2, Module C).

A shrouded impeller photographed under imperfect conditions (tilt, close phone
camera, cluttered desk) shows a *family of concentric ellipses*: the outer rim
of the shroud, the inlet ring (eye), the hub flange and the bore.  This module
finds them robustly, picks the rim and the eye, and provides the affine map
that turns the rim ellipse back into a circle so the downstream analyzer can
work in a (nearly) axial frame.

Pipeline of :func:`find_ellipses`
---------------------------------
1. work copy downscaled to ``work_size`` px (speed), CLAHE, light blur
2. Canny at three threshold pairs -> contours -> candidate arcs (whole contour
   plus sliding windows, so a rim arc that Canny glued to a cable still yields
   a clean sub-arc) -> ``cv2.fitEllipse``; plus region candidates from an Otsu
   mask of the bright shroud (backup when the edge is fragmented)
3. support scoring: fraction of the ellipse perimeter that has an edge pixel
   within ``support_tol`` px *and* whose image gradient is aligned with the
   ellipse normal (the orientation test is what kills desk clutter)
4. non-max suppression of near duplicates, keep the best ``n_refine``
5. refinement at full resolution: for every perimeter sample search along the
   normal for the strongest aligned gradient, refit, three times with a
   shrinking search window (an active-contour step: makes a quarter arc
   converge onto the full rim); rescore at full resolution; final NMS.

Ellipse convention
------------------
:class:`EllipseFit` stores semi-axes ``a >= b`` and ``angle_deg`` = direction
of the major axis in *image* coordinates (x right, y down), such that a point
of the ellipse is ``center + R(angle) @ (a cos t, b sin t)`` with
``R = [[c, -s], [s, c]]`` - the same convention ``cv2.ellipse`` draws with,
so ``e.to_cv()`` can be passed straight to ``cv2.ellipse``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

__all__ = [
    "EllipseFit", "find_ellipses", "pick_impeller_ellipses", "rectify",
    "tilt_from_ellipse", "estimate_perspective_scale", "scale_at_height",
    "transform_ellipse", "invert_affine", "apply_affine", "draw_ellipses",
]


# ---------------------------------------------------------------------------
# data type
# ---------------------------------------------------------------------------
@dataclass
class EllipseFit:
    center: tuple[float, float]          # (cx, cy) px
    axes: tuple[float, float]            # semi-axes (a, b), a >= b
    angle_deg: float                     # major-axis direction, image coords (y down)
    score: float = 0.0                   # chance-corrected perimeter support in [0, 1]
    n_inliers: int = 0                   # perimeter samples with aligned edge within tol
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def a(self) -> float:
        return float(self.axes[0])

    @property
    def b(self) -> float:
        return float(self.axes[1])

    @property
    def aspect(self) -> float:
        """b / a in (0, 1]; 1 = circle."""
        return float(self.axes[1] / self.axes[0]) if self.axes[0] > 0 else 0.0

    @property
    def perimeter(self) -> float:
        a, b = self.axes
        h = ((a - b) / (a + b)) ** 2 if (a + b) > 0 else 0.0
        return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))   # Ramanujan

    @classmethod
    def from_cv(cls, rr: tuple, **kw) -> "EllipseFit":
        """From ``cv2.fitEllipse`` output ``((cx, cy), (w, h), angle)`` (full axes)."""
        (cx, cy), (w, h), ang = rr
        a, b, phi = (w / 2.0, h / 2.0, ang) if w >= h else (h / 2.0, w / 2.0, ang + 90.0)
        return cls((float(cx), float(cy)), (float(a), float(b)), float(phi % 180.0), **kw)

    def to_cv(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """``((cx, cy), (2a, 2b), angle)`` for ``cv2.ellipse`` / ``cv2.fitEllipse`` style."""
        return (self.center, (2.0 * self.axes[0], 2.0 * self.axes[1]), self.angle_deg)

    def points(self, n: int = 360, t: np.ndarray | None = None) -> np.ndarray:
        """(n, 2) float points on the ellipse, parameter t in [0, 2pi)."""
        if t is None:
            t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        c, s = math.cos(math.radians(self.angle_deg)), math.sin(math.radians(self.angle_deg))
        x = self.axes[0] * np.cos(t)
        y = self.axes[1] * np.sin(t)
        return np.stack([self.center[0] + c * x - s * y,
                         self.center[1] + s * x + c * y], axis=1)

    def normals(self, t: np.ndarray) -> np.ndarray:
        """(n, 2) unit outward normals at parameter t."""
        c, s = math.cos(math.radians(self.angle_deg)), math.sin(math.radians(self.angle_deg))
        # tangent in the ellipse frame: (-a sin t, b cos t) -> normal (b cos t, a sin t)
        nx = self.axes[1] * np.cos(t)
        ny = self.axes[0] * np.sin(t)
        n = np.stack([c * nx - s * ny, s * nx + c * ny], axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    def scaled(self, k: float) -> "EllipseFit":
        return EllipseFit((self.center[0] * k, self.center[1] * k),
                          (self.axes[0] * k, self.axes[1] * k), self.angle_deg,
                          self.score, self.n_inliers, dict(self.meta))

    def to_dict(self) -> dict[str, Any]:
        return {"center": [float(self.center[0]), float(self.center[1])],
                "axes": [float(self.axes[0]), float(self.axes[1])],
                "angle_deg": float(self.angle_deg), "score": float(self.score),
                "n_inliers": int(self.n_inliers)}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _prep(gray: np.ndarray, blur_sigma: float = 1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLAHE-normalised, blurred image plus its Sobel gradient (gx, gy) as float32."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    sm = cv2.GaussianBlur(eq, (0, 0), blur_sigma)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return sm, gx, gy


def _canny_levels(sm: np.ndarray, levels=((20, 60), (45, 120), (90, 220))) -> list[np.ndarray]:
    return [cv2.Canny(sm, lo, hi, L2gradient=True) for lo, hi in levels]


def _fit(pts: np.ndarray) -> EllipseFit | None:
    if len(pts) < 6:
        return None
    try:
        rr = cv2.fitEllipse(np.asarray(pts, np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None
    (cx, cy), (w, h), ang = rr
    if not all(np.isfinite([cx, cy, w, h, ang])) or w <= 0 or h <= 0:
        return None
    return EllipseFit.from_cv(rr)


def _plausible(e: EllipseFit, shape: tuple[int, int], a_min: float, a_max: float) -> bool:
    H, W = shape[:2]
    a, b = e.axes
    if not (a_min <= a <= a_max) or b < 0.05 * a:
        return False
    cx, cy = e.center
    return (-0.5 * W <= cx <= 1.5 * W) and (-0.5 * H <= cy <= 1.5 * H)


def _sample_count(e: EllipseFit, spacing: float = 1.0, lo: int = 180, hi: int = 3000) -> int:
    return int(np.clip(e.perimeter / spacing, lo, hi))


_COS_MIN = 0.6                       # |cos| between gradient and ellipse normal
_P_ALIGN = 1.0 - math.degrees(math.acos(_COS_MIN)) / 90.0   # chance of alignment, ~0.59


def _chance_level(dist: np.ndarray, tol: float) -> float:
    """Support a random curve would get by chance: edge density x alignment chance."""
    return float((dist <= tol).mean()) * _P_ALIGN


def _support(e: EllipseFit, dist: np.ndarray, gx: np.ndarray, gy: np.ndarray,
             tol: float, cos_min: float = _COS_MIN, gmin: float = 1.5,
             p_null: float = 0.0) -> tuple[float, int]:
    """Chance-corrected fraction of the perimeter with an aligned edge pixel within *tol* px.

    A perimeter sample is supported when an edge pixel lies within *tol* px and the
    image gradient there is aligned with the ellipse normal (|cos| >= cos_min).
    raw = perimeter fraction covered by *contiguous* supported runs (runs shorter than
    1 % of the perimeter / 8 samples ignored); score = (raw - null) / (1 - null) clipped
    to [0, 1], where *null* is the run-filtered support a random curve would get given
    the image's edge density *p_null*.  Both steps keep texture / noise from producing
    confident false ellipses.  ``n_inliers`` is the raw count of supported samples.
    """
    H, W = dist.shape
    n = _sample_count(e, max(tol, 1.0))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    P = e.points(t=t)
    N = e.normals(t)
    xi = np.rint(P[:, 0]).astype(int)
    yi = np.rint(P[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if inside.sum() < 0.5 * n:
        return 0.0, 0
    xi, yi, N = xi[inside], yi[inside], N[inside]
    d = dist[yi, xi]
    g = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
    gn = np.linalg.norm(g, axis=1)
    cosang = np.abs((g * N).sum(1)) / np.maximum(gn, 1e-6)
    ok = (d <= tol) & (gn >= gmin) & (cosang >= cos_min)
    n_in = int(ok.sum())
    full = np.zeros(n, bool)
    full[np.flatnonzero(inside)] = ok
    # normalise by the full perimeter so ellipses hanging out of the frame are penalised
    min_run = max(8, int(0.01 * n))
    raw = _run_support(full, min_run)
    null = _run_null(p_null, min_run)
    return float(np.clip((raw - null) / max(1.0 - null, 1e-6), 0.0, 1.0)), n_in


def _run_support(ok: np.ndarray, min_run: int) -> float:
    """Fraction of a circular boolean perimeter covered by supported runs of at least
    *min_run* consecutive samples.  Random texture gives short runs and scores ~0; a
    real edge gives runs of dozens of samples."""
    n = len(ok)
    if n == 0 or not ok.any():
        return 0.0
    if ok.all():
        return 1.0
    start = int(np.argmin(ok))                       # rotate so index 0 is unsupported
    c = np.roll(ok, -start)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], c.astype(int), [0]])))
    lens = edges[1::2] - edges[0::2]
    return float(lens[lens >= min_run].sum()) / n


def _run_null(p: float, min_run: int) -> float:
    """Expected run-filtered support of a random curve whose samples are supported
    independently with probability *p* (geometric run lengths)."""
    L = min_run
    return float(p * p ** (L - 1) * (L - (L - 1) * p)) if 0 < p < 1 else 0.0


def _is_dup(e1: EllipseFit, e2: EllipseFit, rel: float = 0.06) -> bool:
    a = max(e1.axes[0], e2.axes[0])
    tol = rel * a
    if abs(e1.axes[0] - e2.axes[0]) > tol or abs(e1.axes[1] - e2.axes[1]) > tol:
        return False
    if math.hypot(e1.center[0] - e2.center[0], e1.center[1] - e2.center[1]) > tol:
        return False
    if e1.aspect < 0.93 and e2.aspect < 0.93:
        dang = abs((e1.angle_deg - e2.angle_deg + 90.0) % 180.0 - 90.0)
        if dang > 15.0:
            return False
    return True


def _nms(cands: list[EllipseFit], rel: float = 0.06) -> list[EllipseFit]:
    keep: list[EllipseFit] = []
    for e in sorted(cands, key=lambda q: q.score, reverse=True):
        if not any(_is_dup(e, k, rel) for k in keep):
            keep.append(e)
    return keep


def _normal_distance(e: EllipseFit, pts: np.ndarray) -> np.ndarray:
    """First-order (Newton) distance of points to the ellipse outline, signed (+ outside)."""
    phi = math.radians(e.angle_deg)
    c, s = math.cos(phi), math.sin(phi)
    u = np.asarray(pts, float) - np.array(e.center)
    ux = c * u[:, 0] + s * u[:, 1]
    uy = -s * u[:, 0] + c * u[:, 1]
    a, b = e.axes
    rho = np.hypot(ux / a, uy / b)
    grad = np.hypot(ux / a ** 2, uy / b ** 2) / np.maximum(rho, 1e-9)
    return (rho - 1.0) / np.maximum(grad, 1e-9)


def _suppress_overlapping(ells: list[EllipseFit], shape: tuple[int, int], tol: float = 3.0,
                          max_shared: float = 0.35) -> list[EllipseFit]:
    """Drop an ellipse when > *max_shared* of its perimeter lies within *tol* px of a
    better-scoring kept ellipse (its support is borrowed from that one)."""
    H, W = shape[:2]
    keep: list[EllipseFit] = []
    for e in sorted(ells, key=lambda q: q.score, reverse=True):
        if keep:
            P = e.points(n=_sample_count(e, 2.0, 180, 1500))
            ok = (P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)
            if ok.sum():
                near = np.zeros(int(ok.sum()), bool)
                for k in keep:
                    near |= np.abs(_normal_distance(k, P[ok])) <= tol
                if near.mean() > max_shared:
                    continue
        keep.append(e)
    return keep


def _arc_candidates(contours, L_min: int, n_pairs: int = 20) -> list[np.ndarray]:
    """Whole contours, sliding sub-arcs (so glued clutter can be shaken off) and
    pairs of the longest arcs (a RANSAC-like step: two opposite arcs of a very
    flat rim fit well together while each alone is nearly a straight line)."""
    out, longest = [], []
    for c in contours:
        pts = c.reshape(-1, 2)
        L = len(pts)
        if L < L_min:
            continue
        out.append(pts)
        longest.append(pts)
        if L >= 2 * L_min:
            win = max(L_min, L // 3)
            step = max(win // 2, 1)
            for s in range(0, L - win + 1, step):
                out.append(pts[s:s + win])
    longest.sort(key=len, reverse=True)
    longest = longest[:n_pairs]
    for i in range(len(longest)):
        for j in range(i + 1, len(longest)):
            out.append(np.vstack([longest[i], longest[j]]))
    return out


def _region_candidates(sm: np.ndarray, a_min: float) -> list[np.ndarray]:
    """Contours of bright blobs (Otsu) - a cheap backup source for the shroud rim."""
    out = []
    _, mask = cv2.threshold(cv2.GaussianBlur(sm, (0, 0), 2.0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        if cv2.contourArea(c) >= math.pi * a_min * a_min * 0.3:
            out.append(c.reshape(-1, 2))
    return out


def _trim_refit(pts: np.ndarray, e: EllipseFit, min_keep: int = 30) -> EllipseFit:
    """Drop points far from *e* (robust MAD rule) and refit once."""
    res = _normal_distance(e, pts)
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    keep = np.abs(res) <= max(1.5, 3.0 * mad)
    if keep.sum() >= max(min_keep, 0.5 * len(pts)):
        r = _fit(pts[keep])
        if r is not None:
            return r
    return e


def _refine(e: EllipseFit, sm: np.ndarray, gx: np.ndarray, gy: np.ndarray,
            windows=(10.0, 5.0, 2.5), cos_min: float = 0.6, conv_tol: float = 0.3) -> EllipseFit | None:
    """Active-contour style refinement.

    For every perimeter sample the projected (normal-aligned) gradient is read
    along the normal within +-window px.  Passes with a window > 4 px take the
    strongest response (pulls a coarse arc fit onto the real edge, iterated
    until the fit stops moving), the small final passes take the strongest
    *peak nearest to the current estimate* (stable: does not jump to a
    neighbouring ring).  Each refit is followed by a robust trim-and-refit so a
    cable crossing the rim does not bias it.
    """
    H, W = sm.shape
    cur = e
    prev_win = None
    for win in windows:
        n = _sample_count(cur, 3.0 if win > 4.0 else 1.5, 240, 2400)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        P = cur.points(t=t)
        N = cur.normals(t)
        steps = np.arange(-win, win + 0.5, 1.0)
        X = P[:, 0:1] + N[:, 0:1] * steps[None, :]
        Y = P[:, 1:2] + N[:, 1:2] * steps[None, :]
        xi = np.rint(X).astype(int)
        yi = np.rint(Y).astype(int)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi_c = np.clip(xi, 0, W - 1)
        yi_c = np.clip(yi, 0, H - 1)
        gxx, gyy = gx[yi_c, xi_c], gy[yi_c, xi_c]
        resp = np.abs(gxx * N[:, 0:1] + gyy * N[:, 1:2])          # projected gradient
        gmag = np.hypot(gxx, gyy)
        aligned = resp >= cos_min * gmag
        resp = np.where(inside & aligned, resp, 0.0)
        gmax = resp.max(axis=1)
        if not np.any(gmax > 0):
            return None
        thr = max(1.5, 0.25 * np.median(gmax[gmax > 0]))
        if win > 4.0:
            k = resp.argmax(axis=1)
        else:
            # nearest local maximum above threshold
            mid = len(steps) // 2
            peak = (resp >= thr)
            peak[:, 1:] &= resp[:, 1:] >= resp[:, :-1]
            peak[:, :-1] &= resp[:, :-1] >= resp[:, 1:]
            cost = np.where(peak, np.abs(np.arange(len(steps)) - mid)[None, :], 10 ** 6)
            k = cost.argmin(axis=1)
            k = np.where(cost[np.arange(n), k] >= 10 ** 6, resp.argmax(axis=1), k)
        best = resp[np.arange(n), k]
        ok = (best >= thr) & (k > 0) & (k < len(steps) - 1)
        if ok.sum() < max(30, 0.15 * n):
            return None
        pts = np.stack([X[np.arange(n), k], Y[np.arange(n), k]], axis=1)[ok]
        if np.ptp(np.unwrap(np.sort(t[ok]))) < math.radians(60):
            return None
        nxt = _fit(pts)
        if nxt is None or not np.isfinite(nxt.axes).all():
            return None
        nxt = _trim_refit(pts, nxt)
        # guard against runaway refits (e.g. jumping to a neighbouring ring)
        if abs(nxt.axes[0] - cur.axes[0]) > 0.25 * cur.axes[0] + 2 * win:
            return None
        moved = max(abs(nxt.axes[0] - cur.axes[0]), abs(nxt.axes[1] - cur.axes[1]),
                    abs(nxt.center[0] - cur.center[0]), abs(nxt.center[1] - cur.center[1]))
        cur = nxt
        if prev_win == win and moved < conv_tol:      # repeated window: converged early
            prev_win = None
            continue
        prev_win = win
    return cur


def _refine_best(e: EllipseFit, sm, gx, gy, dist, support_tol: float,
                 a_min: float, a_max: float, p_null: float = 0.0) -> EllipseFit:
    """Try a conservative and an aggressive refinement schedule, keep the best support."""
    w_big = max(10.0, 0.05 * e.axes[0])
    schedules = (
        (6.0, 3.0, 2.5),                                       # already close: do not wander
        (w_big,) * 8 + (max(4.0, w_big / 3), 2.5),             # far off: walk onto the edge
    )
    cands = [e]
    for sch in schedules:
        r = _refine(e, sm, gx, gy, windows=sch)
        if r is not None and _plausible(r, sm.shape, a_min, a_max):
            cands.append(r)
    for c in cands:
        c.score, c.n_inliers = _support(c, dist, gx, gy, support_tol, p_null=p_null)
    return max(cands, key=lambda q: q.score)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def find_ellipses(img_gray: np.ndarray, *, min_axis_frac: float = 0.15, max_count: int = 6,
                  work_size: int = 800, support_tol: float = 2.0, min_support: float = 0.25,
                  n_refine: int = 20, refine: bool = True,
                  return_debug: bool = False) -> list[EllipseFit] | tuple[list[EllipseFit], dict]:
    """Detect large ellipses (rim, eye ring, hub flange, ...) in a photo.

    Parameters
    ----------
    img_gray : uint8 gray (BGR accepted and converted)
    min_axis_frac : minimum major *semi*-axis as a fraction of ``min(H, W)``
    max_count : maximum number of ellipses returned (sorted by ``score`` desc)
    work_size : longest image side used for the coarse detection
    support_tol : px distance for a perimeter sample to count as supported
    min_support : coarse-stage support threshold
    refine : refine candidates at full resolution (recommended)

    Returns ``list[EllipseFit]`` sorted by ``score`` (chance-corrected fraction of
    the perimeter supported by an aligned edge within ``support_tol`` px, in [0, 1];
    ``n_inliers`` is the raw count of supported perimeter samples).
    """
    gray = _to_gray(np.asarray(img_gray))
    H, W = gray.shape
    a_min_full = min_axis_frac * min(H, W)
    a_max_full = 1.2 * max(H, W)

    # ---- coarse stage --------------------------------------------------
    s = min(1.0, work_size / float(max(H, W)))
    small = cv2.resize(gray, (max(8, int(round(W * s))), max(8, int(round(H * s)))),
                       interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    sm_s, gx_s, gy_s = _prep(small, 1.2)
    edges_levels = _canny_levels(sm_s)
    edge_union = np.maximum.reduce(edges_levels)
    dist_s = cv2.distanceTransform(255 - edge_union, cv2.DIST_L2, 3)
    tol_s = max(1.0, support_tol * s * 1.5)
    p_null_s = _chance_level(dist_s, tol_s)
    a_min_s = a_min_full * s
    a_max_s = a_max_full * s
    L_min = int(max(25, 0.18 * 2 * math.pi * a_min_s))          # ~65 deg of the smallest ellipse

    cand_pts: list[np.ndarray] = []
    for ed in edges_levels:
        cs, _ = cv2.findContours(ed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cand_pts.extend(_arc_candidates(cs, L_min))
    cand_pts.extend(_region_candidates(sm_s, a_min_s))

    coarse: list[EllipseFit] = []
    for pts in cand_pts:
        e = _fit(pts)
        if e is None or not _plausible(e, small.shape, a_min_s, a_max_s):
            continue
        sc, n_in = _support(e, dist_s, gx_s, gy_s, tol_s, p_null=p_null_s)
        if sc >= min_support:
            e.score, e.n_inliers = sc, n_in
            coarse.append(e)
    coarse = _nms(coarse, 0.08)[:n_refine]

    debug: dict[str, Any] = {"n_candidates": len(cand_pts), "n_coarse": len(coarse),
                             "work_scale": s, "chance_level_coarse": p_null_s,
                             "chance_level": p_null_s, "n_final": 0}
    if not coarse:
        return ([], debug) if return_debug else []

    # ---- full-resolution refinement ------------------------------------
    sm_f, gx_f, gy_f = _prep(gray, 1.0 if s < 1.0 else 1.2)
    ed_f = np.maximum.reduce(_canny_levels(sm_f, ((20, 60), (45, 120))))
    dist_f = cv2.distanceTransform(255 - ed_f, cv2.DIST_L2, 3)
    p_null_f = _chance_level(dist_f, support_tol)
    debug["chance_level"] = p_null_f

    final: list[EllipseFit] = []
    for e in coarse:
        ef = e.scaled(1.0 / s) if s < 1.0 else e
        if refine:
            ef = _refine_best(ef, sm_f, gx_f, gy_f, dist_f, support_tol, a_min_full, a_max_full,
                              p_null_f)
        else:
            ef.score, ef.n_inliers = _support(ef, dist_f, gx_f, gy_f, support_tol, p_null=p_null_f)
        ef.meta["chance_level"] = p_null_f
        sc = ef.score
        ef.meta["coarse_score"] = e.score
        if sc >= min_support:
            final.append(ef)
    final = _nms(final, 0.05)
    final = _suppress_overlapping(final, gray.shape, tol=max(3.0, 1.5 * support_tol))
    final.sort(key=lambda q: q.score, reverse=True)
    final = final[:max_count]
    debug["n_final"] = len(final)
    return (final, debug) if return_debug else final


def _concentric(outer: EllipseFit, inner: EllipseFit, ratio_range=(0.3, 0.92)) -> bool:
    ratio = inner.axes[0] / outer.axes[0]
    if not (ratio_range[0] <= ratio <= ratio_range[1]):
        return False
    dc = math.hypot(inner.center[0] - outer.center[0], inner.center[1] - outer.center[1])
    if dc > 0.15 * outer.axes[0]:
        return False
    if abs(inner.aspect - outer.aspect) > 0.12:
        return False
    if outer.aspect < 0.95 and inner.aspect < 0.95:
        dang = abs((inner.angle_deg - outer.angle_deg + 90.0) % 180.0 - 90.0)
        if dang > 25.0:
            return False
    return True


def pick_impeller_ellipses(ellipses: list[EllipseFit], img_shape: tuple[int, ...],
                           *, min_score: float = 0.3, eye_ratio: tuple[float, float] = (0.3, 0.92),
                           ) -> tuple[EllipseFit | None, EllipseFit | None]:
    """Choose (rim, eye) from a list of detections.

    rim : among the well-supported ellipses (score >= min_score, centre inside the
          image) the one maximising ``score * sqrt(a)`` (support-weighted size), with
          a 15 % bonus when it has a roughly concentric smaller ellipse inside it
          (the eye ring / hub flange family).  Fallback: the best-scoring one.
    eye : smaller ellipse (a/rim.a in *eye_ratio*) that is roughly concentric with
          the rim, with a similar aspect ratio (and orientation when both are
          clearly elliptical).  Among several (inlet-ring outer/inner edge, hub
          flange) the best supported wins, with a mild preference for the larger
          one.  ``None`` if nothing qualifies.
    """
    if not ellipses:
        return None, None
    H, W = img_shape[:2]
    good = [e for e in ellipses
            if e.score >= min_score and 0 <= e.center[0] < W and 0 <= e.center[1] < H]
    if not good:
        good = [max(ellipses, key=lambda e: e.score)]
    family = {id(e) for e in good
              if any(_concentric(e, o, eye_ratio) for o in ellipses if o is not e)}
    # support-weighted size: a well-supported rim beats a larger, half-supported stray
    rim = max(good, key=lambda e: e.score * math.sqrt(e.axes[0]) * (1.15 if id(e) in family else 1.0))

    eyes = [(e.score + 0.15 * e.axes[0] / rim.axes[0], e) for e in ellipses
            if e is not rim and e.score >= 0.5 * min_score and _concentric(rim, e, eye_ratio)]
    eye = max(eyes, key=lambda q: q[0])[1] if eyes else None
    return rim, eye


def rectify(img: np.ndarray, rim: EllipseFit, out_size: tuple[int, int] | int | None = None,
            *, margin: float = 0.08, border_value: int = 0
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Affine rectification: send the rim ellipse to a circle of radius ``a``.

    The map stretches the image along the rim's minor axis by ``a / b`` (undoing
    an axis tilt to first order) and moves the rim centre to the centre of the
    output.  Returns ``(img_rect, M, px_per_unit)`` with ``M`` the 2x3 affine
    (input px -> output px; use :func:`apply_affine`/:func:`invert_affine`) and
    ``px_per_unit = a`` = rim radius in output px, i.e. output pixels per unit
    of normalised radius (``px_per_mm = px_per_unit / r2_mm``).

    out_size : (w, h) of the output, an int for a square, or None for a square
               of side ``2 a (1 + margin)`` centred on the rim.
    """
    a, b = rim.axes
    if out_size is None:
        side = int(math.ceil(2.0 * a * (1.0 + margin)))
        out_w = out_h = side
    elif isinstance(out_size, (int, np.integer)):
        out_w = out_h = int(out_size)
    else:
        out_w, out_h = int(out_size[0]), int(out_size[1])
    phi = math.radians(rim.angle_deg)
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]])
    S = np.diag([1.0, a / b])                       # stretch along the minor axis
    A = R @ S @ R.T                                 # rotate into the ellipse frame, stretch, rotate back
    cx, cy = rim.center
    t = np.array([out_w / 2.0, out_h / 2.0]) - A @ np.array([cx, cy])
    M = np.hstack([A, t[:, None]]).astype(np.float64)
    img_rect = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return img_rect, M, float(a)


def tilt_from_ellipse(e: EllipseFit) -> float:
    """Axis tilt [deg] implied by the ellipse foreshortening: ``acos(b / a)``."""
    a, b = e.axes
    if a <= 0:
        return 0.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, b / a)))))


def estimate_perspective_scale(rim: EllipseFit, eye: EllipseFit, rim_z: float, eye_z: float,
                               eye_over_rim_radius: float | None = None, *,
                               rim_radius: float | None = None, eye_radius: float | None = None,
                               ) -> float | None:
    """Camera distance along the axis *above the rim plane* (units of ``rim_z``).

    Pinhole model: apparent radius ``rho = f R / d`` with ``d`` the distance from the
    camera to that plane.  The eye ring sits ``dz = eye_z - rim_z`` closer to the
    camera than the rim, so ``rho_eye / rho_rim = k * D / (D - dz)`` with
    ``k = R_eye / R_rim`` the physical radius ratio -> ``D = dz * q / (q - 1)``,
    ``q = (rho_eye / rho_rim) / k``.  The physical ratio must be supplied either as
    ``eye_over_rim_radius`` or as ``rim_radius`` and ``eye_radius`` (any unit).

    Returns ``None`` when ill-conditioned: ratio unknown, ``dz <= 0``, the eye
    does not appear enlarged (``q <= 1.02`` -> far camera / inconsistent), or the
    solution would put the camera below the eye ring.
    """
    if eye_over_rim_radius is None:
        if rim_radius and eye_radius:
            eye_over_rim_radius = float(eye_radius) / float(rim_radius)
        else:
            return None
    k = float(eye_over_rim_radius)
    dz = float(eye_z) - float(rim_z)
    if dz <= 0 or k <= 0 or rim.axes[0] <= 0:
        return None
    # mean apparent radius (sqrt(a b)) is the perspective-robust size measure
    rho_rim = math.sqrt(rim.axes[0] * rim.axes[1])
    rho_eye = math.sqrt(eye.axes[0] * eye.axes[1])
    q = (rho_eye / rho_rim) / k
    if q <= 1.02:
        return None
    D = dz * q / (q - 1.0)
    if D <= dz:
        return None
    return float(D)


def scale_at_height(camera_dist: float, rim_z: float, z: float) -> float:
    """Relative image scale of a plane at height ``z`` w.r.t. the rim plane for a
    camera ``camera_dist`` above the rim plane (``> 1`` for planes closer to the camera)."""
    d = camera_dist - (z - rim_z)
    if d <= 0:
        raise ValueError("plane is behind the camera")
    return camera_dist / d


# ---------------------------------------------------------------------------
# affine utilities (for Module D overlays)
# ---------------------------------------------------------------------------
def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(M, np.float64))


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to (n, 2) points."""
    P = np.asarray(pts, float).reshape(-1, 2)
    return P @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]


def transform_ellipse(e: EllipseFit, M: np.ndarray) -> EllipseFit:
    """Image of an ellipse under a 2x3 affine (exact: an affine maps ellipses to ellipses)."""
    pts = apply_affine(M, e.points(n=180))
    out = _fit(pts)
    if out is None:                                    # degenerate (should not happen)
        c = apply_affine(M, np.array([e.center]))[0]
        return EllipseFit((float(c[0]), float(c[1])), e.axes, e.angle_deg, e.score, e.n_inliers)
    out.score, out.n_inliers, out.meta = e.score, e.n_inliers, dict(e.meta)
    return out


def draw_ellipses(img: np.ndarray, ellipses: list[EllipseFit], *,
                  colors: list[tuple[int, int, int]] | None = None,
                  thickness: int = 2, label: bool = True) -> np.ndarray:
    """BGR copy of *img* with the ellipses drawn (index + score as label)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(_to_gray(img), cv2.COLOR_GRAY2BGR)
    palette = colors or [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 200, 255),
                         (255, 0, 255), (200, 200, 0)]
    for i, e in enumerate(ellipses):
        col = palette[i % len(palette)]
        (cx, cy), (w, h), ang = e.to_cv()
        cv2.ellipse(out, (int(round(cx)), int(round(cy))),
                    (int(round(w / 2)), int(round(h / 2))), ang, 0, 360, col, thickness, cv2.LINE_AA)
        cv2.drawMarker(out, (int(round(cx)), int(round(cy))), col, cv2.MARKER_CROSS, 16, thickness)
        if label:
            p = e.points(t=np.array([-math.pi / 4]))[0]
            cv2.putText(out, f"{i}:{e.score:.2f}", (int(p[0]) + 4, int(p[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":   # pragma: no cover - manual check
    import sys
    import time
    from pathlib import Path

    for path in sys.argv[1:]:
        img = cv2.imread(path)
        t0 = time.perf_counter()
        ells = find_ellipses(img)
        dt = time.perf_counter() - t0
        rim, eye = pick_impeller_ellipses(ells, img.shape)
        print(f"{path}: {len(ells)} ellipses in {dt:.2f} s")
        for i, e in enumerate(ells):
            tag = " rim" if e is rim else (" eye" if e is eye else "")
            print(f"  [{i}] c=({e.center[0]:.1f},{e.center[1]:.1f}) a={e.a:.1f} b={e.b:.1f} "
                  f"ang={e.angle_deg:.1f} score={e.score:.2f}{tag}")
        if rim is not None:
            print(f"  tilt from rim: {tilt_from_ellipse(rim):.1f} deg")
        over = draw_ellipses(img, ells)
        out = Path(path).with_suffix(".ellipses.png")
        cv2.imwrite(str(out), over)
        print("  overlay ->", out)
