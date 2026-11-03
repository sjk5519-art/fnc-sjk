"""Marker-free beta2 from rim-gap (side) photographs - fanscan v0.3, SPEC3.

Nothing is attached to or drawn on the fan: the outlet blade angle comes from camera angles
alone.  Look at the rim gap (between shroud and backplate) from the side, roughly level with it.
Next to every blade trailing edge (TE) the blade face adjacent to it is visible as a shaded
strip; somewhere along the near half of the rim that face is seen exactly *edge-on* and the
strip flips to the other side of the TE line.

Geometry (math frame, +x = 0, CCW seen from the inlet/+z; camera on the -y axis, elevation
``eps``): for a rim point at azimuth phi the blade exit direction of a CCW-sweeping blade is
``e = (sin(beta2 - phi), cos(beta2 - phi))`` and the face normal ``n = (-cos(beta2 - phi),
sin(beta2 - phi))``; the view direction sees ``v . n = cos(eps) sin(beta2 - phi)``, so the face
is edge-on at ``phi0 = beta2 - 180 deg`` on the near side, at image offset
``x0 = -r2 cos(beta2)`` from the rim centre (``+r2 cos(beta2)`` for CW blades):

    |x0| / a_rim = cos(beta2)        and the side of the centre gives the handedness.

Between the TEs the transition is interpolated from a *local* measurement at every TE: with a
camera above the gap the lit backplate top is visible as a band under every receding surface,
and the band is 0 high at the foot of a TE on its face side and grows along the face.  The
face bottom edge leaves the TE foot at the screen angle

    psi_i = atan( sin(eps) * cot(beta2 - phi_i) )          (far camera, 2-D blade)

which crosses +-90 deg at phi0 (vertical: the face recedes straight away) and 0 where the face
is seen flat on.  Both the *sign* (face left / right of the TE: the side with the smaller band)
and the *magnitude* of psi_i are measured and the 1-parameter model is fitted over all TEs of
the near half (sin(eps) = b / a of the rim arcs).  Writing the model with B in (0, 180) covers
both hands at once: ``B < 90`` -> CCW with beta2 = B, ``B > 90`` -> CW with beta2 = 180 - B
(a horizontal mirror maps one onto the other).  Perspective replaces phi_i by phi_i + alpha_i
with alpha_i the ray angle to that TE (from the rim model's ``c = r2 / D``); when c is
unknown a far camera is assumed and the confidence interval widened.

Refinement by synthesis (optional, default on): a numpy blade mesh (:mod:`fanscan.fastmesh`)
of the measured impeller (gap height, blade count, handedness, TE phase) is rendered through a
camera fitted to the detected rim arcs with ``render3d.render`` for candidate beta2 (coarse
2 deg sweep + fine 0.5 deg pass); both gap strips are rectified to the same (x, z) grid and
the normalised cross-correlation of their horizontal-gradient / band patterns is maximised.
This handles perspective, elevation, blade curvature and occlusion exactly.

Per-image pipeline
------------------
1. gap-band hypotheses in the central columns (own detector: bright-above edge on top, a
   horizontal edge below, straight vertical edges between them; ``shrouded.analyze_rim_image``;
   a RANSAC over equal-height vertical edge runs) -> for each, the rim arcs by a coarse-to-fine
   search of the shared shape (xc, a) with per-arc sagittas: edge energy summed along the gap
   top arc (bright-above polarity), the backplate arc, the far arc of the shroud rim top
   (pins a; weighted by b / a because the shroud cone hides it when the view is flat) and the
   vertical rim silhouette at xc +- a, with off-edge samples penalised so arcs are not
   stretched; snapping to the edges and a robust joint refit.  The hypothesis whose arcs are
   best supported wins; the inner edge of each parallel pair (rim top / underside, backplate
   top / bottom) is taken as the gap boundary.
2. gap strip rectified to (x, v in [0, 1]) between the arcs; TEs = straight (up to +-12 deg
   sheared, for a close camera) vertical edges spanning the strip, on a contrast-normalised
   strip, fitted as the pitch-periodic family ``x = xc + A sin(phi) / (1 - c cos(phi))`` with
   ``A = a sqrt(1 - c^2)`` tied to the arcs (>= 4 TEs also give the perspective term c)
3. per-TE band heights left / right -> face side; band slope on the face side -> psi_i
4. fit B -> beta2, handedness, x0; confidence from the residual, the TE count on each side
   of the transition and the camera-distance uncertainty
5. synthesis refinement: a pinhole camera (distance, height, pitch, yaw, focal length, gap
   height) is fitted to the two arcs, the TE phase is aligned over a full pitch by the TE-line
   pattern, and beta2 is swept at reduced resolution for the hand found in step 4 (both hands
   when that is weak: the better NCC decides the hand).  The synthesis result replaces the
   sign-change estimate when its peak is clear.

Several images (different azimuths, no registration needed) are combined by an
inverse-variance weighted mean; a disagreement > 8 deg lowers the confidence and warns.

Conventions: image frame y down, shroud *up* in the photo, camera outside the rim.  With
the shroud up, screen +x is +theta (CCW seen from the inlet): a CCW impeller shows its blade
bodies on the *left* of the TEs at the image centre (``x0`` left of the centre), a CW one on
the right.  ``handedness`` is reported as seen from the inlet, the same convention as
``shrouded.analyze_shrouded``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from . import fastmesh, render3d
from .params import ImpellerParams, SHROUDED_PARAMS_FACTORY
from .pose import EllipseFit

__all__ = ["RimGapResult", "analyze_rim_gap", "draw_rim_gap_overlay", "analyze_rim_gap_image",
           "synthesize_strip", "rectify_strip"]

_WORK_WIDTH = 1400
_MIN_W, _MIN_H = 100, 60
_STRIP_ROWS = 48               # rows of the rectified gap strip
_DISAGREE_DEG = 8.0            # multi-image disagreement that costs confidence
_PRIOR_BETA2 = 40.0
_PRIOR_BETA1 = 30.0
_C_PRIOR = 0.15                # rim model perspective term r2 / D when it cannot be fitted
_SYNTH_COARSE = (15.0, 75.0, 2.0)
_SYNTH_FINE_STEP = 0.5
_SYNTH_FINE_HALF = 3.0
_SYNTH_SCALE_PX = 520          # render width used for the synthesis sweeps


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------
@dataclass
class RimGapResult:
    beta2: float
    confidence: float
    handedness: str                        # 'ccw' | 'cw' | 'undetermined'
    x0_px: float | None                    # edge-on offset from the rim centre (image px, first image)
    te_x: list[float]                      # detected TE positions (image px, first image)
    te_widths: list[float]                 # signed face widths / band-slope measure per TE (px, >0 = +x side)
    rim: EllipseFit | None                 # backplate rim arc as an ellipse (first image)
    gap: dict                              # gap top / bottom lines at the centre column, px (first image)
    per_image: list[dict]
    method: str                            # 'sign-change' | 'synthesis' | 'prior'
    debug: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"beta2": self.beta2, "confidence": self.confidence, "handedness": self.handedness,
                "x0_px": self.x0_px, "te_x": list(self.te_x), "te_widths": list(self.te_widths),
                "rim": None if self.rim is None else self.rim.to_dict(),
                "gap": dict(self.gap), "method": self.method, "warnings": list(self.warnings),
                "per_image": [{k: v for k, v in d.items() if not k.startswith("_")} for d in self.per_image]}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_gray(img: Any) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.shape[2] == 3 else a[..., 0]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(img.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _arc_y(x: np.ndarray, xc: float, a: float, b: float, yc: float) -> np.ndarray:
    """Near (lower) arc of an axis-aligned ellipse: y at the centre column = yc."""
    u = np.clip((np.asarray(x, float) - xc) / max(a, 1e-9), -1.0, 1.0)
    return yc + b * (np.sqrt(1.0 - u * u) - 1.0)


# ---------------------------------------------------------------------------
# 1. rim arcs
# ---------------------------------------------------------------------------
def _sample(E: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of E at float (x, y) (0 outside)."""
    H, W = E.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    ok = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    x0c, y0c = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    v = (E[y0c, x0c] * (1 - fx) * (1 - fy) + E[y0c, x0c + 1] * fx * (1 - fy)
         + E[y0c + 1, x0c] * (1 - fx) * fy + E[y0c + 1, x0c + 1] * fx * fy)
    return np.where(ok, v, 0.0)


def _snap_track(E: np.ndarray, y: np.ndarray, win: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Move every track point to the strongest edge within +-win rows (parabolic sub-pixel);
    returns (y_refined, strength)."""
    H, W = E.shape
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    offs = np.arange(-win, win + 1)
    rows = np.clip(yi[None, :] + offs[:, None], 0, H - 1)
    vals = E[rows, np.arange(W)[None, :]]
    k = vals.argmax(axis=0)
    best = rows[k, np.arange(W)].astype(float)
    st = vals[k, np.arange(W)]
    km = np.clip(k, 1, len(offs) - 2)
    y0, y1, y2 = vals[km - 1, np.arange(W)], vals[km, np.arange(W)], vals[km + 1, np.arange(W)]
    den = y0 - 2 * y1 + y2
    sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    return best + np.clip(sub, -1, 1), st


def _vertical_runs(gx_abs: np.ndarray, gy_abs: np.ndarray, *, min_len_frac: float = 0.04,
                   max_len_frac: float = 0.6, min_fill: float = 0.6) -> np.ndarray:
    """Straight vertical edge segments: per column the extent (first .. last row) of the rows
    where |gx| is at least 0.35 of the column maximum and dominates |gy|, accepted when at least
    *min_fill* of that extent is filled (a trailing-edge line keeps its contrast across the
    backplate band and the face shading, but may be interrupted).  Returns (n, 4) rows of
    (x, y_start, y_end, strength) sorted by x; neighbouring columns of one line are merged."""
    H, W = gx_abs.shape
    floor = 0.15 * float(np.percentile(gx_abs, 99.5))
    cmax = gx_abs.max(axis=0)
    M = (gx_abs > np.maximum(0.35 * cmax[None, :], floor)) & (gx_abs > 1.2 * gy_abs)
    M = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8)) > 0
    out = []
    for x in range(W):
        col = M[:, x]
        if cmax[x] <= floor or not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        # best extent: grow from the longest run over neighbouring runs while the fill stays high
        k = int((ends - starts).argmax())
        lo, hi = k, k
        while True:
            grown = False
            for cand in ((lo - 1, hi), (lo, hi + 1)):
                a_, b_ = cand
                if a_ < 0 or b_ >= len(starts):
                    continue
                ext = ends[b_] - starts[a_]
                fill = float(np.sum(ends[a_:b_ + 1] - starts[a_:b_ + 1])) / max(1, ext)
                if fill >= min_fill and ext <= max_len_frac * H:
                    lo, hi = a_, b_
                    grown = True
                    break
            if not grown:
                break
        y0, y1 = int(starts[lo]), int(ends[hi] - 1)
        L = y1 - y0 + 1
        if min_len_frac * H <= L <= max_len_frac * H:
            seg = gx_abs[y0:y1 + 1, x]
            out.append((x, y0, y1, float(np.median(seg))))
    if not out:
        return np.zeros((0, 4))
    R = np.array(out, float)
    merged = []
    i = 0
    while i < len(R):
        j = i
        while j + 1 < len(R) and R[j + 1, 0] - R[j, 0] <= 2 and abs(R[j + 1, 1] - R[i, 1]) < 0.15 * (R[i, 2] - R[i, 1]) \
                and abs(R[j + 1, 2] - R[i, 2]) < 0.15 * (R[i, 2] - R[i, 1]):
            j += 1
        grp = R[i:j + 1]
        merged.append(grp[int(grp[:, 3].argmax())])
        i = j + 1
    return np.array(merged)


def _te_ransac(runs: np.ndarray, H: int, W: int, *, min_inliers: int = 3
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Trailing-edge runs = the largest set of equal-height vertical segments whose top and
    bottom endpoints lie on two smooth (parabolic) arcs.  RANSAC over run triples; returns
    (inlier runs, top parabola coefficients, bottom parabola coefficients) or None."""
    if len(runs) < min_inliers:
        return None
    runs = runs[runs[:, 3] >= 0.25 * float(np.percentile(runs[:, 3], 95))]
    n = len(runs)
    if n < min_inliers:
        return None
    if n > 80:
        runs = runs[np.argsort(-runs[:, 3])[:80]]
        n = len(runs)
    x, yt, yb, st = runs.T
    L = yb - yt
    idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])
    if len(idx) == 0:
        return None
    xi, xj, xk = x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]
    span = np.maximum.reduce([xi, xj, xk]) - np.minimum.reduce([xi, xj, xk])
    sep = np.minimum.reduce([np.abs(xi - xj), np.abs(xj - xk), np.abs(xi - xk)])
    Lm = np.median(np.stack([L[idx[:, 0]], L[idx[:, 1]], L[idx[:, 2]]], 1), axis=1)
    L_ok = (np.abs(L[idx] - Lm[:, None]) <= 0.2 * Lm[:, None]).all(axis=1)
    ok = (span >= 0.08 * W) & (sep >= 0.015 * W) & L_ok
    idx = idx[ok]
    if len(idx) == 0:
        return None
    # parabola through 3 points: scaled x for conditioning
    xs_ = (x - 0.5 * W) / W
    A = np.stack([np.ones((len(idx), 3)), xs_[idx], xs_[idx] ** 2], axis=2)          # (m, 3, 3)
    try:
        ct = np.linalg.solve(A, yt[idx][:, :, None])[:, :, 0]                       # (m, 3)
        cb = np.linalg.solve(A, yb[idx][:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        return None
    # the arcs bulge downward at the centre (y max at xc): c2 <= 0 (allow a little noise)
    hg = Lm[ok]
    conc = (ct[:, 2] <= 0.02 * H) & (cb[:, 2] <= 0.02 * H)
    Pt = ct[:, 0][:, None] + ct[:, 1][:, None] * xs_[None, :] + ct[:, 2][:, None] * xs_[None, :] ** 2
    Pb = cb[:, 0][:, None] + cb[:, 1][:, None] * xs_[None, :] + cb[:, 2][:, None] * xs_[None, :] ** 2
    tol = np.maximum(3.0, 0.05 * hg)[:, None]
    inl = (np.abs(Pt - yt[None, :]) <= tol) & (np.abs(Pb - yb[None, :]) <= tol) & conc[:, None]
    # inliers must be distinct lines: count columns, score by capped strength
    cap = float(np.percentile(st, 75))
    score = (inl * np.minimum(st, cap)[None, :]).sum(axis=1) * np.sqrt(inl.sum(axis=1).clip(1))
    best = int(score.argmax())
    sel = inl[best]
    if sel.sum() < min_inliers:
        return None
    members = runs[sel]
    # refit the parabolas on the inliers
    M = np.stack([np.ones(sel.sum()), xs_[sel], xs_[sel] ** 2], 1)
    ct_f = np.linalg.lstsq(M, yt[sel], rcond=None)[0]
    cb_f = np.linalg.lstsq(M, yb[sel], rcond=None)[0]
    return members, ct_f, cb_f


def _fit_arcs(xs: np.ndarray, yn: np.ndarray, yf: np.ndarray, yb: np.ndarray, wn: np.ndarray,
              wf: np.ndarray, wb: np.ndarray, q0: list[float]) -> tuple[list[float], float]:
    """Joint robust fit of the gap top arc (yn), the far arc of the shroud rim top (yf, same
    ellipse shifted up by d_far) and the backplate near arc yb (shared xc, a).
    q = (xc, a, b_t, y0_t, b_b, y0_b, d_far) -> (q, rms)."""
    def resid(q):
        xc, a, bt, y0t, bb, y0b, df = q
        u = np.clip((xs - xc) / max(a, 1e-9), -1, 1)
        root = np.sqrt(1 - u * u)
        rn = (y0t + bt * root - yn) * wn
        rf = (y0t - bt * root - df - yf) * wf
        rb = (y0b + bb * root - yb) * wb
        return np.concatenate([rn, rf, rb])

    xc0, a0, bt0, y0t0, bb0, y0b0, df0 = q0
    lo = [xc0 - 0.1 * a0, 0.85 * a0, 0.0, y0t0 - 25, 0.0, y0b0 - 25, 0.0]
    hi = [xc0 + 0.1 * a0, 1.18 * a0, 1.2 * a0, y0t0 + 25, 1.2 * a0, y0b0 + 25, 30.0]
    q0 = np.clip(q0, lo, hi)
    sol = least_squares(resid, q0, bounds=(lo, hi), loss="soft_l1", f_scale=1.5)
    r = sol.fun[np.concatenate([wn, wf, wb]) > 0]
    rms = float(np.sqrt(np.mean(np.minimum(np.abs(r), 6.0) ** 2))) if len(r) else 9.9
    return [float(v) for v in sol.x], rms


class _ArcEvaluator:
    """Edge maps and the coarse arc objective of :func:`_find_arcs` (kept separate so the
    objective can be inspected)."""

    def __init__(self, gray, runs, gy_abs, gy_neg, gx_abs, gmag=None):
        H, W = gray.shape
        self.H, self.W = H, W
        self.xr, self.yt_r, self.yb_r = runs[:, 0], runs[:, 1], runs[:, 2]
        self.hg0 = hg0 = float(np.median(self.yb_r - self.yt_r))
        x_mid = float(np.median(self.xr))
        rt, rb = int(round(np.median(self.yt_r))), int(round(np.median(self.yb_r)))
        cols = slice(int(max(0, x_mid - 0.1 * W)), int(min(W, x_mid + 0.1 * W)))
        if gmag is None:
            gmag = np.sqrt(gx_abs ** 2 + gy_abs ** 2)
        g_top = np.where(gy_neg > 0.3 * gmag, gmag, 0.0)
        n_top = max(1e-6, float(np.percentile(g_top[max(0, rt - 3):rt + 4, cols].max(axis=0), 60)))
        n_bot = max(1e-6, float(np.percentile(gmag[max(0, rb - 3):rb + 4, cols].max(axis=0), 60)))
        E_top = np.clip(g_top / n_top, 0, 1.5)
        E_abs = np.clip(gmag / n_bot, 0, 1.5)
        band = slice(int(max(0, rt - 2 * hg0)), int(min(H, rb + 1.5 * hg0)))
        tau = 0.3
        self.E_abs = E_abs - max(tau, 1.5 * float(np.mean(E_abs[band])))
        self.E_top = E_top - max(tau, 1.5 * float(np.mean(E_top[band])))
        n_v = max(1e-6, float(np.percentile(gx_abs[rt:rb + 1, cols], 97)))
        E_v = np.clip(gx_abs / n_v, 0, 1.5)
        E_v = E_v - max(tau, 1.5 * float(np.mean(E_v[band])))
        # smooth horizontally so the term varies smoothly with a (the grids are coarse)
        self.E_v = 0.3 * cv2.GaussianBlur(cv2.dilate(E_v, np.ones((1, 5), np.uint8)), (0, 0), max(2.0, 0.008 * W), 0.1)
        self.far_offsets = np.array([0.0, 6.0, 12.0])    # far arc = rim TOP edge, a few px above
        self.n_sil = int(max(8, 0.8 * hg0))
        spread = float(self.xr.max() - self.xr.min())
        self.a_min = max(1.3 * hg0, 0.05 * W, 0.55 * spread)
        self.a_max = max(self.a_min * 1.05, min(0.62 * W, 9.0 * hg0))
        self.us = np.linspace(-0.97, 0.97, 97)
        self.root = np.sqrt(1 - self.us * self.us)
        self.jit = np.array([-4.0, 0.0, 4.0])

    def sil_term(self, xc, a, y0s):
        H, W = self.H, self.W
        rows = np.clip(np.round(y0s)[..., None] + np.arange(self.n_sil)[None, :], 0, H - 1).astype(int)
        xl = np.clip(np.round(xc - a), 0, W - 1).astype(int)
        xr_ = np.clip(np.round(xc + a), 0, W - 1).astype(int)
        return self.E_v[rows, xl[..., None]].sum(axis=-1) + self.E_v[rows, xr_[..., None]].sum(axis=-1)

    def terms(self, xc, a, bt, y0t, bb, y0b):
        """(top arc, far arc (weighted), silhouette, bottom arc) score terms of one model."""
        x = xc + self.us * a
        Yn = y0t + bt * self.root
        Yf = y0t - bt * self.root
        w_far = float(np.clip((bt / a - 0.12) / 0.1, 0.0, 1.0))
        t_top = float(_sample(self.E_top, x, Yn).mean()) * a
        t_far = w_far * max(float(_sample(self.E_abs, x, Yf - d).mean()) for d in self.far_offsets) * a
        t_sil = float(self.sil_term(np.array([xc]), np.array([a]), np.array([y0t]))[0])
        t_bot = float(_sample(self.E_abs, x, y0b + bb * self.root).mean()) * a
        return t_top, t_far, t_sil, t_bot

    def evaluate(self, xcs, as_, bts, bbs, fine=False):
        xr, yt_r, yb_r, us, root, jit = self.xr, self.yt_r, self.yb_r, self.us, self.root, self.jit
        best = None
        for xc in xcs:
            ur = np.clip((xr - xc) / as_[:, None], -1, 1)
            rr = np.sqrt(1 - ur * ur)
            for j, a in enumerate(as_):
                x = xc + us * a
                bt = bts * a if not fine else bts
                bb = bbs * a if not fine else bbs
                y0t = np.median(yt_r[None, :] - bt[:, None] * rr[j][None, :], axis=1)
                y0b = np.median(yb_r[None, :] - bb[:, None] * rr[j][None, :], axis=1)
                y0t = (y0t[:, None] + jit[None, :]).ravel()
                y0b = (y0b[:, None] + jit[None, :]).ravel()
                btj = np.repeat(bt, len(jit))
                bbj = np.repeat(bb, len(jit))
                X = np.broadcast_to(x[None, :], (len(btj), len(us)))
                Yn = y0t[:, None] + btj[:, None] * root[None, :]
                Yf = y0t[:, None] - btj[:, None] * root[None, :]
                w_far = np.clip((btj / a - 0.12) / 0.1, 0.0, 1.0)
                far = np.max([_sample(self.E_abs, X, Yf - d).mean(axis=1) for d in self.far_offsets], axis=0)
                sc_t = (_sample(self.E_top, X, Yn).mean(axis=1) + w_far * far) * a
                sc_t = sc_t + self.sil_term(np.full_like(y0t, xc), np.full_like(y0t, a), y0t)
                Yb = y0b[:, None] + bbj[:, None] * root[None, :]
                sc_b = _sample(self.E_abs, np.broadcast_to(x[None, :], (len(bbj), len(us))), Yb).mean(axis=1) * a
                it, ib = int(sc_t.argmax()), int(sc_b.argmax())
                tot = float(sc_t[it] + sc_b[ib])
                if best is None or tot > best[0]:
                    best = (tot, float(xc), float(a), float(btj[it]), float(y0t[it]), float(bbj[ib]), float(y0b[ib]))
        return best


def _find_arcs(gray: np.ndarray, runs: np.ndarray, gy_abs: np.ndarray, gy_neg: np.ndarray,
               gx_abs: np.ndarray, debug: dict[str, Any], gmag: np.ndarray | None = None,
               curvature: tuple[float, float] | None = None, xc_prior: float | None = None
               ) -> dict[str, Any]:
    """Rim arcs (work px) from a set of trailing-edge runs (x, y_top, y_bot, strength).

    Every TE line ends on the two gap arcs, so for any arc shape (xc, a, b) the arc's row
    offset is the median over the runs of the implied centre row: the search is over the shape
    only.  Score = edge energy summed along the gap top arc (bright-above polarity) and the
    backplate arc (any polarity), the far arc of the shroud rim top (pins a; hidden by the
    shroud cone when the ellipse is flat, hence weighted by b / a) and the vertical rim
    silhouette at x = xc +- a.  Off-edge samples cost tau so arcs are not stretched.  Then
    snapping to the edges and a robust joint refit."""
    H, W = gray.shape
    ev = _ArcEvaluator(gray, runs, gy_abs, gy_neg, gx_abs, gmag)
    evaluate, a_min, a_max = ev.evaluate, ev.a_min, ev.a_max
    E_top, E_abs = ev.E_top, ev.E_abs
    xr, yt_r, yb_r = ev.xr, ev.yt_r, ev.yb_r
    hg0 = ev.hg0
    if xc_prior is not None and 0.2 * W < xc_prior < 0.8 * W:
        xcs = np.arange(xc_prior - 0.08 * W, xc_prior + 0.08 * W + 1e-6, max(2.0, 0.01 * W))
    else:
        xcs = np.arange(0.34 * W, 0.66 * W + 1e-6, max(2.0, 0.02 * W))
    as_ = np.exp(np.linspace(math.log(a_min), math.log(a_max), 18))
    if curvature is not None:
        # b = kappa a^2 (kappa = -2 c2 from the parabola fits, px^-1): search a and b / (kappa a^2)
        kt, kb = curvature
        facs = np.array([0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5])
        best = None
        for a in as_:
            bts = np.clip(kt * a * a * facs, 0.005 * a, 0.9 * a) / a
            bbs = np.clip(kb * a * a * facs, 0.005 * a, 0.9 * a) / a
            cand = evaluate(xcs, np.array([a]), bts, bbs)
            if cand is not None and (best is None or cand[0] > best[0]):
                best = cand
    else:
        ratios = np.linspace(0.02, 0.8, 16)
        best = evaluate(xcs, as_, ratios, ratios)
    _, xc, a, bt, y0t, bb, y0b = best
    xcs2 = xc + np.linspace(-0.02 * W, 0.02 * W, 9)
    as2 = a * np.linspace(0.9, 1.1, 21)
    bts2 = np.linspace(max(0.005 * a, bt - 0.06 * a), bt + 0.06 * a, 9)
    bbs2 = np.linspace(max(0.005 * a, bb - 0.06 * a), bb + 0.06 * a, 9)
    best = evaluate(xcs2, as2, bts2, bbs2, fine=True)
    _, xc, a, bt, y0t, bb, y0b = best
    coarse = (xc, a, bt, y0t, bb, y0b)
    q = [xc, a, bt, y0t, bb, y0b, 0.0]
    xs = np.arange(W, dtype=float)
    rms = 9.9
    for win_px in (4, 3):
        xc, a, bt, y0t, bb, y0b, df = q
        u = np.clip((xs - xc) / a, -1, 1)
        rt_ = np.sqrt(1 - u * u)
        yn, sn = _snap_track(E_top, y0t + bt * rt_, win=win_px)
        yf, sf = _snap_track(E_abs, y0t - bt * rt_ - df, win=win_px + 2)
        yb, sb = _snap_track(E_abs, y0b + bb * rt_, win=win_px)
        inside = np.abs(u) < 0.97
        wn = (inside & (sn > 0.05)).astype(float)
        wf = (inside & (sf > 0.05)).astype(float) * 0.7 * float(np.clip((bt / a - 0.12) / 0.1, 0.0, 1.0))
        wb = (inside & (sb > 0.05)).astype(float)
        if wn.sum() < 0.15 * a or wb.sum() < 0.15 * a:
            break
        q, rms = _fit_arcs(xs, yn, yf, yb, wn, wf, wb, q)
    xc, a, bt, y0t, bb, y0b, df = q
    u = np.clip((xs - xc) / max(a, 1e-6), -1, 1)
    rt_ = np.sqrt(1 - u * u)
    yn, sn = _snap_track(E_top, y0t + bt * rt_, win=3)
    yb, sb = _snap_track(E_abs, y0b + bb * rt_, win=3)
    inside = np.abs(u) < 0.97
    wn = (inside & (np.abs(yn - (y0t + bt * rt_)) < 2.5) & (sn > 0.05)).astype(float)
    wb = (inside & (np.abs(yb - (y0b + bb * rt_)) < 2.5) & (sb > 0.05)).astype(float)
    frac_t = float(wn.sum() / max(1.0, 2 * 0.97 * a))
    frac_b = float(wb.sum() / max(1.0, 2 * 0.97 * a))
    # agreement of the TE runs with the arcs (px)
    ur = np.clip((xr - xc) / a, -1, 1)
    te_res = float(np.median(np.abs(y0t + bt * np.sqrt(1 - ur * ur) - yt_r))
                   + np.median(np.abs(y0b + bb * np.sqrt(1 - ur * ur) - yb_r)))
    debug["arcs"] = {"xc": xc, "a": a, "b_top": bt, "b_bot": bb, "yc_top": y0t + bt, "yc_bot": y0b + bb,
                     "y0_shroud": y0t, "d_far": df, "rms": rms, "support_top": frac_t, "support_bot": frac_b,
                     "coarse": coarse, "a_range": (a_min, a_max), "n_runs": int(len(runs)), "te_run_resid": te_res}
    return {"xc": float(xc), "a": float(a), "b": float(bb), "b_top": float(bt), "b_bot": float(bb),
            "yc_top": float(y0t + bt), "yc_bot": float(y0b + bb), "y0_shroud": float(y0t), "rms": rms,
            "support": min(frac_t, frac_b), "te_run_resid": te_res,
            "_track_top": yn, "_track_bot": yb, "_w_top": wn, "_w_bot": wb, "_runs": runs}


# ---------------------------------------------------------------------------
# 2. rectified strip + TEs
# ---------------------------------------------------------------------------
def rectify_strip(gray: np.ndarray, arcs: dict[str, Any], *, rows: int = _STRIP_ROWS,
                  u_max: float = 0.985, inset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample the gap between the two arcs onto a (rows, n_cols) grid.

    Row 0 is the top (shroud underside edge), row ``rows - 1`` the bottom (backplate top edge);
    ``inset`` (fraction of the gap) trims both edges.  Returns (strip float32, x columns)."""
    xc, a = arcs["xc"], arcs["a"]
    bt, bb = arcs.get("b_top", arcs["b"]), arcs.get("b_bot", arcs["b"])
    x0 = int(math.ceil(xc - u_max * a))
    x1 = int(math.floor(xc + u_max * a))
    W = gray.shape[1]
    x0, x1 = max(0, x0), min(W - 1, x1)
    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    yt = _arc_y(xs, xc, a, bt, arcs["yc_top"])
    yb = _arc_y(xs, xc, a, bb, arcs["yc_bot"])
    v = (np.arange(rows) + 0.5) / rows
    v = inset + (1.0 - 2.0 * inset) * v
    Y = yt[None, :] + v[:, None] * (yb - yt)[None, :]
    X = np.broadcast_to(xs[None, :], Y.shape)
    return _bilinear(gray, X, Y), xs


def _refine_gap_edges(gray: np.ndarray, arcs: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    """Resolve the parallel-edge ambiguity of each arc (shroud rim top vs underside edge,
    backplate top vs bottom edge): the gap boundary is the *inner* edge of a close pair.  The
    row profile of |gy| of a generously rectified strip shows the pair as two peaks; take the
    lowest peak near the top arc and the highest peak near the bottom arc."""
    hg = arcs["yc_bot"] - arcs["yc_top"]
    ext = 0.15
    a2 = dict(arcs)
    a2["yc_top"] = arcs["yc_top"] - ext * hg
    a2["yc_bot"] = arcs["yc_bot"] + ext * hg
    rows = int(np.clip(hg * (1 + 2 * ext), 40, 200))
    strip, xs = rectify_strip(gray.astype(np.float32), a2, rows=rows, u_max=0.9)
    _, gy = _strip_gradients(strip)
    prof = gaussian_filter1d(np.abs(gy).mean(axis=1), 0.8)
    px_per_row = hg * (1 + 2 * ext) / rows
    pk, _ = find_peaks(prof, prominence=0.08 * float(prof.max() + 1e-9))
    if len(pk) == 0:
        return arcs
    out = dict(arcs)
    info = {}
    r_top = ext * hg / px_per_row
    r_bot = (ext * hg + hg) / px_per_row
    win_in, win_out = 0.14 * hg / px_per_row, 0.05 * hg / px_per_row
    top_c = [q for q in pk if r_top - win_out <= q <= r_top + win_in]
    bot_c = [q for q in pk if r_bot - win_in <= q <= r_bot + win_out]
    if top_c:
        ref = max(prof[q] for q in top_c)
        q = max(q for q in top_c if prof[q] >= 0.25 * ref)
        out["yc_top"] = float(a2["yc_top"] + (q + 0.5) * px_per_row)
        info["d_top"] = out["yc_top"] - arcs["yc_top"]
    if bot_c:
        ref = max(prof[q] for q in bot_c)
        q = min(q for q in bot_c if prof[q] >= 0.25 * ref)
        out["yc_bot"] = float(a2["yc_top"] + (q + 0.5) * px_per_row)
        info["d_bot"] = out["yc_bot"] - arcs["yc_bot"]
    if out["yc_bot"] - out["yc_top"] < 0.6 * hg:
        return arcs
    debug["gap_refine"] = info
    return out


def _strip_gradients(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sm = cv2.GaussianBlur(strip, (0, 0), 1.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def _rim_x(phi: np.ndarray, xc: float, A: float, c: float) -> np.ndarray:
    """Rim model: screen x of the rim point at azimuth phi from the camera direction."""
    return xc + A * np.sin(phi) / (1.0 - c * np.cos(phi))


def _find_tes(strip: np.ndarray, xs: np.ndarray, Z: int, arcs: dict[str, Any], c_prior: float,
              debug: dict[str, Any]) -> dict[str, Any]:
    """Trailing edges = straight vertical edges spanning the strip, as a pitch-periodic family on
    the rim model x = xc + A sin(phi) / (1 - c cos(phi))."""
    rows, n = strip.shape
    # local contrast normalisation: on a real (cream-on-cream) impeller the TE lines are faint
    # next to the silhouette edges, which are excluded from every statistic below (|u| > 0.93)
    eq = cv2.createCLAHE(3.0, (4, 16)).apply(np.clip(strip, 0, 255).astype(np.uint8)).astype(np.float32)
    gx, _ = _strip_gradients(eq)
    ax = np.abs(gx)
    u_cols = (xs - arcs["xc"]) / max(arcs["a"], 1e-6)
    core = np.abs(u_cols) <= 0.96
    # column profile: mean |gx| over the middle 80 % of rows, and the fraction of rows that carry an
    # edge; a close camera makes the (3-D vertical) TE lines converge, so the profile is taken
    # over a few shear angles (+-12 deg) and the best response per column is kept
    r0, r1 = int(0.1 * rows), int(0.9 * rows)
    hg_px = arcs["yc_bot"] - arcs["yc_top"]
    px_per_row = hg_px / rows
    thr_row = 0.35 * float(np.percentile(ax[r0:r1][:, core], 99)) if core.any() else 0.35 * float(np.percentile(ax[r0:r1], 99))
    Vs = np.zeros(n)
    for ang in (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0):
        shear = math.tan(math.radians(ang)) * px_per_row          # strip px of x per row
        M = np.array([[1.0, -shear, shear * (rows / 2.0)], [0.0, 1.0, 0.0]], np.float32)
        axs = cv2.warpAffine(ax, M, (n, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) if ang != 0.0 else ax
        V = gaussian_filter1d(axs[r0:r1].mean(axis=0), 1.0)
        span = gaussian_filter1d((axs[r0:r1] > thr_row).mean(axis=0), 1.0)
        Vs = np.maximum(Vs, V * np.clip(span / 0.5, 0.0, 1.0))   # straight full-height edges only
    Vs = np.where(core, Vs, 0.0)                                 # the silhouette edges are not TEs
    # candidates: generous (faint cream-on-cream TE lines next to strong silhouette-side edges);
    # the pitch-periodic family fit below does the selection
    pk, _ = find_peaks(Vs, prominence=0.04 * float(Vs.max() + 1e-9), distance=max(2, int(0.012 * n)))
    pk = pk[Vs[pk] > 0.04 * Vs.max()]
    if len(pk) > 24:
        pk = np.sort(pk[np.argsort(Vs[pk])[::-1][:24]])
    cand = xs[pk].astype(float)
    cand_v = Vs[pk] / max(float(Vs.max()), 1e-9)
    xc, A = arcs["xc"], arcs["a"]
    pitch = 2 * math.pi / Z
    ks = np.arange(-int(math.pi / 2 / pitch) - 1, int(math.pi / 2 / pitch) + 2)
    tol = max(3.0, 0.012 * n)
    best = None
    if len(cand) >= 2:
        for c in (c_prior,) if c_prior is not None else (0.05, 0.15, 0.3):
            for Af in np.linspace(0.94, 1.06, 7):
                for ph0 in np.arange(0.0, pitch, math.radians(0.5)):
                    phis = ph0 + ks * pitch
                    sel = np.abs(phis) <= math.radians(85)
                    # silhouette half-width a = A / sqrt(1 - c^2)
                    xp = _rim_x(phis[sel], xc, Af * A * math.sqrt(1 - c * c), c)
                    dist = np.abs(cand[:, None] - xp[None, :])
                    match = dist.min(axis=0) <= tol
                    used = np.unique(dist.argmin(axis=0)[match])
                    if len(used) < 2:
                        continue
                    res = float(np.sqrt(np.mean(dist.min(axis=0)[match] ** 2)))
                    score = len(used) + 1.0 * float(np.sqrt(cand_v[used]).sum()) - 0.5 * res / tol
                    if best is None or score > best[0]:
                        best = (score, used, Af * A * math.sqrt(1 - c * c), ph0, c, res)
    if best is None:
        return {"te_x": np.array([]), "te_phi": np.array([]), "te_v": np.array([]), "model": None,
                "candidates": cand, "V": Vs}
    _, used, A_f, ph0, c_f, res = best
    te = cand[used]
    tv = cand_v[used]
    kk = np.round((np.arcsin(np.clip((te - xc) / (A_f / math.sqrt(1 - c_f * c_f)), -1, 1)) - ph0) / pitch)

    n_te = len(te)
    # A is tied to the arcs' half-width (A = a sqrt(1 - c^2)); with >= 3 TEs the spacing pattern
    # then determines the perspective term c
    def resid(q):
        xc_, p0_, c_ = q
        return _rim_x(p0_ + kk * pitch, xc_, A * math.sqrt(1 - c_ * c_), c_) - te

    lo = [xc - 0.03 * n, ph0 - pitch, 0.02]
    hi = [xc + 0.03 * n, ph0 + pitch, 0.6]
    q0 = [xc, ph0, c_f]
    if n_te < 3:
        lo[2], hi[2] = c_f - 1e-6, c_f + 1e-6
        lo[0], hi[0] = xc - 1e-6, xc + 1e-6
    try:
        sol = least_squares(resid, np.clip(q0, lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=tol)
        xc_f, ph0_f, c_fit = [float(v) for v in sol.x]
        res = float(np.sqrt(np.mean(sol.fun ** 2)))
    except ValueError:
        xc_f, ph0_f, c_fit = xc, ph0, c_f
    A_fit = A * math.sqrt(1 - c_fit * c_fit)
    phis = ph0_f + kk * pitch
    order = np.argsort(te)
    model = {"xc": xc_f, "A": A_fit, "phi0_deg": math.degrees(ph0_f), "c": c_fit, "resid_px": res,
             "c_fitted": bool(n_te >= 4 and res < 0.5 * tol), "k": kk[order].tolist()}
    debug["te"] = {"n_candidates": int(len(cand)), "n_te": int(n_te), "model": model}
    return {"te_x": te[order], "te_phi": phis[order], "te_v": tv[order], "model": model,
            "candidates": cand, "V": Vs}


# ---------------------------------------------------------------------------
# 3. band heights / face side / slope per TE
# ---------------------------------------------------------------------------
def _band_profile(strip: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Height (fraction of the gap) of the backplate band at every column = the lowest strong
    horizontal edge in the lower part of the strip; 0 when the surface reaches the backplate
    edge.  A horizontal edge shared by most columns at the same row (the backplate's second,
    parallel edge when the arc landed on the outer one) is taken as the band base instead.
    Also returns the per-column edge strength used."""
    rows, n = strip.shape
    ag = np.abs(gy)
    lo, hi = int(0.2 * rows), rows - max(4, int(0.05 * rows))
    thr = 0.3 * float(np.percentile(ag[int(0.1 * rows):int(0.9 * rows)], 99))
    base = rows - 1

    def lowest_hit(hi_row):
        sub = ag[lo:hi_row]
        hit = sub > thr
        any_hit = hit.any(axis=0)
        k = np.argmax(hit[::-1], axis=0)
        idx = hi_row - 1 - k
        return np.where(any_hit, idx, -1)

    idx = lowest_hit(hi)
    valid = idx >= 0
    if valid.sum() > 0.5 * n:
        vals, counts = np.unique(idx[valid], return_counts=True)
        # a shared row (+-1) over > 45 % of the columns is a rim edge, not the band
        for v, c in sorted(zip(vals, counts), key=lambda t: -t[1])[:3]:
            share = np.sum(np.abs(idx[valid] - v) <= 1) / n
            if share > 0.45 and v > 0.6 * rows:
                base = int(v) - 2
                idx = lowest_hit(base)
                break
    h = np.where(idx >= 0, (base - idx) / max(1, base), 0.0)
    h = median_filter(np.clip(h, 0.0, 1.0), size=3)
    strength = ag[lo:hi].max(axis=0) / max(thr, 1e-6)
    return h, strength


def _te_measurements(strip: np.ndarray, xs: np.ndarray, tes: dict[str, Any], arcs: dict[str, Any],
                     debug: dict[str, Any]) -> list[dict[str, Any]]:
    """Per TE: band height left / right, face side, band slope on the face side -> psi."""
    gx, gy = _strip_gradients(strip)
    rows, n = strip.shape
    h, _ = _band_profile(strip, gy)
    te_x = tes["te_x"]
    out = []
    d = max(3, int(0.01 * n))
    gap_px = (arcs["yc_bot"] - arcs["yc_top"])              # gap height at the centre column (px)
    for i, x in enumerate(te_x):
        j = int(round(x - xs[0]))
        if j < 0 or j >= n:
            continue
        # neighbours: limit the measuring windows to half the way to the next TE
        left_lim = int(round(te_x[i - 1] - xs[0])) if i > 0 else 0
        right_lim = int(round(te_x[i + 1] - xs[0])) if i + 1 < len(te_x) else n - 1
        wl = int(max(2 * d, min(0.15 * (j - left_lim), 0.06 * n)))
        wr = int(max(2 * d, min(0.15 * (right_lim - j), 0.06 * n)))
        hl = h[max(0, j - 3 * d):max(1, j - d)]
        hr = h[j + d:j + 3 * d]
        h_left = float(np.median(hl)) if len(hl) else 0.0
        h_right = float(np.median(hr)) if len(hr) else 0.0
        # the TE line itself is a full-height edge: the face side is where the band starts at 0
        side = 0
        if abs(h_left - h_right) > 0.04:
            side = 1 if h_right < h_left else -1
        # local column-scale: gap height at this column in px (arcs are vertical offsets -> same as centre)
        hg = gap_px
        res = {"x": float(x), "phi_deg": math.degrees(float(tes["te_phi"][i])), "h_left": h_left,
               "h_right": h_right, "side": side, "strength": float(tes["te_v"][i])}
        # slope of the band on the face side (robust linear fit of h over a short window)
        for s, lim in ((1, wr), (-1, wl)):
            seg = h[j + d:j + d + lim] if s == 1 else h[max(0, j - d - lim):max(0, j - d)][::-1]
            if len(seg) < 3:
                res[f"slope_{'r' if s == 1 else 'l'}"] = None
                continue
            # stop at the first jump (occluder)
            dh = np.abs(np.diff(seg))
            jumps = np.where(dh > 0.12)[0]
            k_end = int(jumps[0]) + 1 if len(jumps) and jumps[0] >= 2 else len(seg)
            seg = seg[:max(3, min(k_end, len(seg)))]
            t = np.arange(len(seg), dtype=float) + d           # px from the TE
            # least-squares line through the TE foot over the rising part (h < 0.5)
            use = seg < 0.5
            m = float(np.sum(seg[use] * t[use]) / max(1e-9, np.sum(t[use] ** 2))) if use.sum() >= 2 else 0.0
            res[f"slope_{'r' if s == 1 else 'l'}"] = m * hg   # px per px
        if side == 1:
            m = res["slope_r"]
        elif side == -1:
            m = res["slope_l"]
        else:
            m = None
        res["psi_deg"] = None if m is None else float(side * math.degrees(math.atan(abs(m))))
        # signed face width proxy: band-free run on the face side (px)
        if side != 0:
            seg = h[j + d:j + d + (wr if side == 1 else 0)] if side == 1 else h[max(0, j - d - wl):max(0, j - d)][::-1]
            run = int(np.argmax(seg > 0.08)) if np.any(seg > 0.08) else len(seg)
            res["width_px"] = float(side * (run + d))
        else:
            res["width_px"] = 0.0
        out.append(res)
    debug["band_h"] = h
    return out


# ---------------------------------------------------------------------------
# 4. edge-on transition fit
# ---------------------------------------------------------------------------
def _ray_angle(phi: np.ndarray, c: float) -> np.ndarray:
    """Horizontal angle of the ray from the camera to the rim point at phi (rad, + = right)."""
    return np.arctan2(c * np.sin(phi), 1.0 - c * np.cos(phi))


def _fit_transition(meas: list[dict[str, Any]], k_elev: float, c: float, debug_key: str,
                    debug: dict[str, Any]) -> dict[str, Any] | None:
    """Fit psi_i = atan(k cot(B - phi_i - alpha_i)) over B in (0, 180) with k ~ sin(eps)."""
    ok = [m for m in meas if m["psi_deg"] is not None and m["side"] != 0]
    if len(ok) < 2:
        return None
    phi = np.radians([m["phi_deg"] for m in ok]) - math.pi / 2       # math frame (camera on -y)
    alpha = _ray_angle(phi + math.pi / 2, c)
    psi = np.radians([m["psi_deg"] for m in ok])
    side = np.array([m["side"] for m in ok], float)
    w = np.array([0.5 + 0.5 * min(1.0, m["strength"]) for m in ok])
    w *= np.where(np.abs(np.degrees([m["phi_deg"] for m in ok])) < 75, 1.0, 0.4)
    B_grid = np.radians(np.arange(2.0, 178.01, 0.25))
    ks = k_elev * np.array([0.6, 0.8, 1.0, 1.25, 1.6])
    best = None
    for k in ks:
        arg = B_grid[:, None] - (phi + alpha)[None, :]
        model = np.arctan(k / np.tan(arg))
        # the measured psi carries the side sign; the model must agree in sign: penalise sign errors hard
        r = psi[None, :] - model
        cost = (w[None, :] * r ** 2).sum(axis=1) / w.sum()
        sign_err = (np.sign(model) != side[None, :]) & (np.abs(model) > math.radians(3))
        cost = cost + (w[None, :] * sign_err * (math.radians(60) ** 2)).sum(axis=1) / w.sum()
        i = int(cost.argmin())
        if best is None or cost[i] < best[0]:
            best = (float(cost[i]), float(B_grid[i]), float(k), cost)
    cost_min, B, k, cost = best
    rms = math.degrees(math.sqrt(cost_min))
    # curvature of the cost around the minimum -> interval where cost <= min + (5 deg)^2
    B_deg = np.degrees(B_grid)
    thr = cost_min + math.radians(6.0) ** 2
    inside = B_deg[cost <= thr]
    lo, hi = (float(inside.min()), float(inside.max())) if len(inside) else (B_deg[0], B_deg[-1])
    n_pos = int(np.sum(side > 0))
    n_neg = int(np.sum(side < 0))
    out = {"B_deg": math.degrees(B), "k": k, "rms_deg": rms, "interval_deg": (lo, hi), "n_te": len(ok),
           "n_left": n_neg, "n_right": n_pos}
    debug[debug_key] = out
    return out


def _beta_from_B(B: float) -> tuple[float, str]:
    if B < 90.0:
        return B, "ccw"
    return 180.0 - B, "cw"


# ---------------------------------------------------------------------------
# per-image driver
# ---------------------------------------------------------------------------
def _gap_seed(gray: np.ndarray, Z: int) -> dict[str, Any]:
    """shrouded.analyze_rim_image on *gray* (work px) -> central-row gap and silhouette."""
    from .shrouded import analyze_rim_image

    rr = analyze_rim_image(gray, Z, work_width=gray.shape[1])
    s = rr["scale"]
    sl, sr, complete = rr["silhouette"]
    return {"y_top": rr["gap"][0], "y_bot": rr["gap"][1], "sil": (sl, sr, complete),
            "te_x_seed": rr["te_x"], "model_seed": rr["model"], "scale": s}


def _gap_seeds(gx_abs: np.ndarray, gy_abs: np.ndarray, gy_neg: np.ndarray, n_max: int = 3
               ) -> list[tuple[float, float, float]]:
    """Candidate gap bands (y_top, y_bot, score) from the central columns: a bright-above
    horizontal edge on top, a horizontal edge at the bottom, and between them straight vertical
    edges (trailing edges) spanning most of the band, with little vertical-edge energy just
    outside the band."""
    H, W = gx_abs.shape
    cols = slice(int(0.35 * W), int(0.65 * W))
    gx_c = gx_abs[:, cols]
    nt = float(np.percentile(gy_abs[:, cols], 99)) + 1e-6
    T = gaussian_filter1d(gy_neg[:, cols].mean(axis=1), 1.0) / nt
    Bm = gaussian_filter1d(gy_abs[:, cols].mean(axis=1), 1.0) / nt
    thr_v = 0.25 * float(np.percentile(gx_c, 99)) + 1e-6
    strong = gx_c > thr_v                                   # (H, ncols)
    cs = np.concatenate([np.zeros((1, strong.shape[1])), np.cumsum(strong, axis=0)], 0)
    Vrow = gaussian_filter1d(gx_c.mean(axis=1), 1.0) / (float(np.percentile(gx_c, 99)) + 1e-6)
    pk_t, _ = find_peaks(T, prominence=0.05, distance=3)
    pk_b, _ = find_peaks(Bm, prominence=0.05, distance=3)
    pk_t = [q for q in pk_t if T[q] > 0.15]
    pk_b = [q for q in pk_b if Bm[q] > 0.15]
    cands = []
    for yt in pk_t:
        for yb in pk_b:
            hg = yb - yt
            if not (0.04 * H <= hg <= 0.5 * H):
                continue
            r0, r1 = int(yt + 0.12 * hg), int(yb - 0.12 * hg)
            if r1 - r0 < 4:
                continue
            frac = (cs[r1] - cs[r0]) / (r1 - r0)             # per column: fraction of band rows with a vertical edge
            n_te = float(np.sum(frac > 0.7))                 # straight full-height vertical edges
            v_in = float(Vrow[r0:r1].mean())
            out_rows = np.concatenate([Vrow[max(0, yt - int(0.4 * hg)):max(1, yt - 2)],
                                       Vrow[yb + 3:min(H, yb + int(0.4 * hg) + 3)]])
            v_out = float(out_rows.mean()) if len(out_rows) else 0.0
            sc = (min(T[yt], 1.0) + min(Bm[yb], 1.0) + 1.5 * min(1.0, n_te / 3.0)
                  + 1.0 * np.clip((v_in - v_out) / (v_in + v_out + 1e-6), -1, 1))
            cands.append((float(yt), float(yb), float(sc)))
    cands.sort(key=lambda c: -c[2])
    out: list[tuple[float, float, float]] = []
    for c in cands:
        if all(abs(c[0] - o[0]) > 0.05 * H or abs(c[1] - o[1]) > 0.05 * H for o in out):
            out.append(c)
        if len(out) >= n_max:
            break
    return out


def _count_full_height_edges(gray: np.ndarray, arcs: dict[str, Any]) -> int:
    """Number of distinct straight vertical edges spanning >= 80 % of the gap band between
    the arcs (|u| < 0.9)."""
    strip, xs = rectify_strip(gray.astype(np.float32), arcs, rows=32, u_max=0.9)
    gx, _ = _strip_gradients(strip)
    ax = np.abs(gx)
    rows = strip.shape[0]
    thr = 0.35 * float(np.percentile(ax[2:-2], 99)) + 1e-6
    span = (ax[2:-2] > thr).mean(axis=0)
    pk, _ = find_peaks(gaussian_filter1d(span, 1.0), height=0.8, distance=max(3, int(0.01 * len(span))))
    return int(len(pk))


def analyze_rim_gap_image(img: np.ndarray, blade_count: int, *, work_width: int = _WORK_WIDTH,
                          c_hint: float | None = None) -> dict[str, Any]:
    """Sign-change analysis of ONE side photo.  Returns a dict (image px) with ``beta2``,
    ``handedness``, ``confidence``, ``B_deg``, ``x0_px``, ``te`` (per-TE measurements), ``arcs``,
    ``gap``, ``strip`` (rectified gap, float32), ``warnings``.  Raises ValueError when no gap
    band is found."""
    Z = int(blade_count)
    if Z < 2:
        raise ValueError("blade_count must be >= 2")
    gray0 = _to_gray(img)
    H0, W0 = gray0.shape
    if H0 < _MIN_H or W0 < _MIN_W:
        raise ValueError(f"rim image too small ({W0}x{H0}); need at least {_MIN_W}x{_MIN_H}")
    s = min(1.0, work_width / max(H0, W0))
    gray = cv2.resize(gray0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray0
    H, W = gray.shape
    debug: dict[str, Any] = {"scale": s}
    warnings: list[str] = []
    eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    sm = cv2.GaussianBlur(eq, (0, 0), 1.2).astype(np.float32)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gy_abs = np.abs(gy)
    gy_neg = np.maximum(0.0, -gy)
    gx_abs = np.abs(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0)
    runs = _vertical_runs(gx_abs, gy_abs)
    hyps: list[tuple[np.ndarray, tuple[float, float] | None, float | None]] = []
    rs = _te_ransac(runs, H, W)
    if rs is not None:
        members, ct, cb = rs
        # parabola y = c0 + c1 u + c2 u^2 with u = (x - W/2) / W: curvature kappa = -2 c2 / W^2,
        # vertex x = W/2 - c1 W / (2 c2)
        kt, kb = max(0.0, -2 * ct[2] / W ** 2), max(0.0, -2 * cb[2] / W ** 2)
        c2 = 0.5 * (ct[2] + cb[2])
        xv = 0.5 * W - 0.5 * (ct[1] + cb[1]) * W / (2 * c2) if c2 < -1e-9 else None
        hyps.append((members, (kt, kb) if min(kt, kb) > 0 else None, xv))
        debug["te_ransac"] = {"n_inliers": int(len(members)), "x": members[:, 0].tolist(),
                              "curvature": (kt, kb), "x_vertex": xv}
    # band seeds (central columns): the top candidates of the own detector + analyze_rim_image's
    own = _gap_seeds(gx_abs, gy_abs, gy_neg, 5)
    for yt, yb, sc in own:
        hyps.append((np.array([[0.5 * W, yt, yb, 1.0]]), None, None))
    seed_rim = None
    try:
        sd = _gap_seed(gray, Z)
        sl, sr, _ = sd["sil"]
        if all(abs(sd["y_top"] - yt) > 0.02 * H or abs(sd["y_bot"] - yb) > 0.02 * H for yt, yb, _ in own):
            hyps.append((np.array([[0.5 * (sl + sr), sd["y_top"], sd["y_bot"], 1.0]]), None, None))
        seed_rim = sd
    except ValueError as exc:
        if not hyps:
            raise ValueError(f"rim image: no rim-gap band found ({exc})") from exc
    debug["band_seeds"] = own
    best_arcs, best_dbg = None, None
    for k, (members, curv, xv) in enumerate(hyps):
        dbg_q: dict[str, Any] = {}
        try:
            arcs_q = _find_arcs(gray, members, gy_abs, gy_neg, gx_abs, dbg_q, curvature=curv, xc_prior=xv)
        except (ValueError, np.linalg.LinAlgError):
            continue
        d = dbg_q["arcs"]
        pen = 1.0 + 0.2 * d["te_run_resid"] if len(members) >= 3 else 1.0
        # the gap band holds straight vertical edges (trailing edges) spanning its full height
        n_full = _count_full_height_edges(gray, arcs_q)
        quality = (d["support_top"] + d["support_bot"]) ** 2 * arcs_q["a"] / pen * (0.3 + min(n_full, 4) / 4.0)
        arcs_q["n_full_edges"] = n_full
        arcs_q["_hyp"] = k
        debug.setdefault("hypotheses", []).append({"k": k, "seed_rows": (float(members[0, 1]), float(members[0, 2])),
                                                   "n_members": int(len(members)), "a": arcs_q["a"], "xc": arcs_q["xc"],
                                                   "n_full": n_full,
                                                   "support": (d["support_top"], d["support_bot"]), "quality": quality,
                                                   "yc": (arcs_q["yc_top"], arcs_q["yc_bot"])})
        if best_arcs is None or quality > best_arcs[0]:
            best_arcs, best_dbg = (quality, arcs_q), dbg_q
    if best_arcs is None:
        raise ValueError("rim image: rim arcs not found")
    arcs = best_arcs[1]
    debug.update(best_dbg)
    debug["n_hypotheses"] = len(hyps)
    debug["hypothesis_used"] = arcs["_hyp"]
    seed = {"model_seed": None if seed_rim is None else seed_rim.get("model_seed"),
            "te_x_seed": [] if seed_rim is None else seed_rim.get("te_x_seed", [])}
    if arcs["support"] < 0.25:
        warnings.append(f"rim arcs weakly supported ({arcs['support']:.2f} of the near half)")
    arcs = _refine_gap_edges(gray, arcs, debug)
    n_rows = int(np.clip(arcs["yc_bot"] - arcs["yc_top"], 32, 128))
    strip, xs = rectify_strip(gray.astype(np.float32), arcs, rows=n_rows)
    c_prior = c_hint
    if c_prior is None and seed.get("model_seed") is not None and len(seed.get("te_x_seed", [])) >= 5:
        c_prior = float(np.clip(seed["model_seed"]["c"], 0.03, 0.5))
    tes = _find_tes(strip, xs, Z, arcs, c_prior, debug)
    meas = _te_measurements(strip, xs, tes, arcs, debug)
    k_elev = float(np.clip(arcs["b"] / max(arcs["a"], 1e-6), 0.03, 0.95))
    c_fit = tes["model"]["c"] if tes["model"] is not None else _C_PRIOR
    c_known = bool(tes["model"] is not None and (tes["model"]["c_fitted"] or c_hint is not None))
    fit = _fit_transition(meas, k_elev, c_fit, "fit", debug)
    out: dict[str, Any] = {
        "scale": s, "size": (W0, H0), "arcs": {k: (v / s if k != "support" and k != "rms" else v)
                                                for k, v in arcs.items() if not k.startswith("_")},
        "gap": {"top": arcs["yc_top"] / s, "bottom": arcs["yc_bot"] / s, "height_px": (arcs["yc_bot"] - arcs["yc_top"]) / s},
        "te": [{**m, "x": m["x"] / s, "width_px": m["width_px"] / s} for m in meas],
        "te_x": [float(x / s) for x in tes["te_x"]], "n_te": int(len(tes["te_x"])),
        "rim_model": tes["model"], "k_elev": k_elev, "c": c_fit, "c_known": c_known,
        "_strip": strip, "_strip_xs": xs / s, "_gray_work": gray, "_arcs_work": arcs, "_tes_work": tes,
        "_meas_work": meas, "warnings": warnings, "debug": debug,
    }
    if fit is None:
        out.update({"beta2": None, "handedness": "undetermined", "confidence": 0.0, "B_deg": None,
                    "x0_px": None, "sigma_deg": None})
        warnings.append("fewer than 2 trailing edges with a measurable face side")
        return out
    B = fit["B_deg"]
    beta2, hand = _beta_from_B(B)
    # camera-distance uncertainty: refit with c at the ends of the plausible range when c is unknown
    spread = 0.0
    if not c_known:
        alts = [_fit_transition(meas, k_elev, c_alt, f"fit_c{c_alt:.2f}", debug) for c_alt in (0.0, 0.35)]
        Bs = [f["B_deg"] for f in alts if f is not None]
        if Bs:
            spread = max(abs(b - B) for b in Bs)
    sigma = math.hypot(0.5 * (fit["interval_deg"][1] - fit["interval_deg"][0]) / 2.0, spread)
    n_side = min(fit["n_left"], fit["n_right"])
    conf = 1.0 / (1.0 + (fit["rms_deg"] / 8.0) ** 2)
    conf *= 0.55 if n_side == 0 else (0.85 if n_side == 1 else 1.0)
    conf *= min(1.0, 0.35 + 0.2 * fit["n_te"])
    conf *= 1.0 / (1.0 + spread / 6.0)
    if abs(B - 90.0) < 3.0:
        hand = "undetermined"
        conf *= 0.6
    if n_side == 0:
        warnings.append("edge-on transition not bracketed by trailing edges (all faces on one side)")
    # x0 (image px): edge-on where B - phi - alpha(phi) = 180 deg (phi in the math frame)
    phi_grid = np.radians(np.linspace(-179.0, -1.0, 357))
    g = np.degrees(phi_grid + _ray_angle(phi_grid + math.pi / 2, c_fit)) - (B - 180.0)
    j = int(np.argmin(np.abs(g)))
    x0 = float(_rim_x(np.array([phi_grid[j] + math.pi / 2]), 0.0, arcs["a"] * math.sqrt(1 - c_fit ** 2), c_fit)[0]) / s \
        if tes["model"] is not None else -arcs["a"] * math.cos(math.radians(B)) / s
    out.update({"beta2": float(beta2), "handedness": hand, "confidence": float(np.clip(conf, 0.0, 1.0)),
                "B_deg": float(B), "x0_px": x0, "sigma_deg": float(max(sigma, 1.0)), "fit": fit})
    return out


# ---------------------------------------------------------------------------
# 5. synthesis refinement
# ---------------------------------------------------------------------------
class _OffsetCamera(render3d.Camera):
    """render3d.Camera with a free principal point (sub-window renders)."""

    def __init__(self, *a, cx=None, cy=None, **kw):
        super().__init__(*a, **kw)
        self._cx, self._cy = cx, cy

    @property
    def cx(self) -> float:          # type: ignore[override]
        return (self.size[0] - 1) / 2.0 if self._cx is None else float(self._cx)

    @property
    def cy(self) -> float:          # type: ignore[override]
        return (self.size[1] - 1) / 2.0 if self._cy is None else float(self._cy)


def _synth_params(Z: int, b2_over_r2: float, beta1: float, beta2: float,
                  base: ImpellerParams | None = None) -> ImpellerParams:
    """Impeller for the synthesis: factory shape with the measured blade count, gap height
    (b2 / r2) and the candidate angles (the hub is short so it stays out of the gap view)."""
    import dataclasses
    p = base or SHROUDED_PARAMS_FACTORY()
    r2 = p.r2
    b2 = float(np.clip(b2_over_r2 * r2, 0.05 * r2, 0.6 * r2))
    beta2 = float(np.clip(beta2, 5.0, 150.0))
    beta1 = float(np.clip(beta1, 5.0, 150.0))
    return dataclasses.replace(p, blade_count=int(Z), blade_angle_outlet=beta2, blade_angle_inlet=beta1,
                               blade_width=b2, blade_width_inlet=max(p.b1, b2 + p.shroud_cone_height),
                               inner_diameter=min(p.inner_diameter, 0.8 * p.eye_diameter),
                               hub_height=20.0, meta={})


def _fit_camera(arcs: dict[str, Any], p: ImpellerParams, size: tuple[int, int],
                full_size: tuple[int, int], c_init: float = _C_PRIOR, c_sigma: float = 0.1
                ) -> tuple[_OffsetCamera, dict[str, Any]]:
    """Pinhole camera (work px) whose projected backplate rim circle (r2 + 2, z = tb) and
    shroud underside circle (r2, z = tb + b2) reproduce the two detected arcs.

    The camera sits at (0, -D, z_cam) (impeller frame), looks with pitch (down) and yaw, has
    the principal point at the centre of the full work image and focal length f: 5 parameters
    (D, z_cam, pitch, yaw, f).  A close camera sees the lower backplate arc with a larger
    sagitta than the shroud arc, which is what pins D."""
    xc, a = arcs["xc"], arcs["a"]
    bt, bb = arcs["b_top"], arcs["b_bot"]
    r2 = p.r2
    tb = float(p.backplate_thickness)
    z_bot = tb
    W_full, H_full = full_size
    cx0, cy0 = (W_full - 1) / 2.0, (H_full - 1) / 2.0
    phi = np.linspace(-math.radians(100), math.radians(100), 81)
    near = np.abs(phi) <= math.radians(80)              # beyond the tangent points the circle is on the far arc

    def make(q):
        D, zc, pitch, yaw, f = q[:5]
        pos = np.array([0.0, -D, zc])
        fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
        return _OffsetCamera(pos, pos + fwd, np.array([0.0, 0.0, 1.0]), f, size, cx=cx0, cy=cy0)

    def resid(q):
        cam = make(q)
        b2 = q[5]
        circles = ((r2, z_bot + b2, bt, arcs["yc_top"]), (r2 + fastmesh.BACKPLATE_RIM_MM, z_bot, bb, arcs["yc_bot"]))
        out = []
        for r, z, b_, yc_ in circles:
            P = np.stack([r * np.sin(phi), -r * np.cos(phi), np.full_like(phi, z)], 1)
            pr = cam.project(P)
            if not np.all(pr[:, 2] > 1.0):
                return np.full(2 * (len(phi) + 2), 1e3)
            # every projected rim point must lie on the observed arc (vertical distance), and the
            # projected extent must match the arc's extent (x_c +- a)
            inside = near & (np.abs(pr[:, 0] - xc) <= 0.98 * a)
            dy = np.where(inside, pr[:, 1] - _arc_y(pr[:, 0], xc, a, b_, yc_), 0.0)
            out.append(dy * (0.6 if z > z_bot else 1.0))          # the backplate arc is the cleaner one
            out.append(np.array([3.0 * (pr[:, 0].max() - (xc + a)), 3.0 * (pr[:, 0].min() - (xc - a))]))
        out.append(np.array([8.0 * (r2 / q[0] - c_init) / c_sigma]))   # prior on the perspective term
        return np.concatenate(out)

    c0 = float(np.clip(c_init, 0.03, 0.6))
    f0 = a * math.sqrt(1 - c0 * c0) / c0
    D0 = r2 / c0
    eps0 = math.asin(min(0.95, max(0.03, 0.5 * (bt + bb) / a)))
    yaw0 = math.atan((xc - cx0) / f0)
    gap_px = arcs["yc_bot"] - arcs["yc_top"]
    b2_0 = float(np.clip(gap_px * (1 - c0) / (a * math.sqrt(1 - c0 * c0) * math.cos(eps0)) * r2, 0.05 * r2, 0.6 * r2))
    # pitch so that the gap centre row lands where it was seen
    y_gap = 0.5 * (arcs["yc_top"] + arcs["yc_bot"]) - 0.5 * (bt + bb)       # ellipse-centre row of the gap
    pitch0 = eps0 - math.atan((y_gap - cy0) / f0)
    zc0 = z_bot + 0.5 * b2_0 + D0 * math.tan(eps0)
    best = None
    for Dk in (D0, 0.5 * D0, 2.0 * D0):
        q0 = [Dk, zc0 * Dk / D0, pitch0, yaw0, f0 * Dk / D0, b2_0]
        lo = [r2 / 0.7, -2000.0, math.radians(-30.0), math.radians(-40.0), 0.1 * f0, 0.05 * r2]
        hi = [r2 / 0.01, 5000.0, math.radians(80.0), math.radians(40.0), 20.0 * f0, 0.6 * r2]
        try:
            sol = least_squares(resid, np.clip(q0, lo, hi), bounds=(lo, hi),
                                x_scale=[0.2 * D0, 50.0, 0.1, 0.1, 0.2 * f0, 0.1 * r2])
        except ValueError:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    sol = best
    cam = make(sol.x)
    D, zc, pitch, yaw, f, b2 = [float(v) for v in sol.x]
    # elevation of the near rim point as seen from the camera
    el = math.degrees(math.atan2(zc - (z_bot + 0.5 * b2), D - r2))
    info = {"D_mm": D, "z_cam_mm": zc, "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw),
            "focal_px": f, "elevation_deg": el, "c": float(r2 / D), "b2_mm": b2, "b2_over_r2": b2 / r2,
            "rms_px": float(np.sqrt(np.mean(sol.fun ** 2)))}
    return cam, info


def _strip_feature(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features compared between photo and synthesis: the blurred horizontal-gradient
    magnitude of the strip (TE lines, face boundaries) and the band profile."""
    gx, gy = _strip_gradients(strip)
    f = cv2.GaussianBlur(np.abs(gx), (0, 0), 3.0, 1.5)
    h, _ = _band_profile(strip, gy)
    return f.astype(np.float32), gaussian_filter1d(h, 3.0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def _scaled(cam: _OffsetCamera, arcs: dict[str, Any], k: float) -> tuple[_OffsetCamera, dict[str, Any]]:
    """Camera and arcs for a render at scale k (cheaper sweeps)."""
    cam2 = _OffsetCamera(cam.position, cam.target, cam.up, cam.focal_px * k,
                         (max(8, int(round(cam.size[0] * k))), max(8, int(round(cam.size[1] * k)))),
                         cx=(cam.cx + 0.5) * k - 0.5, cy=(cam.cy + 0.5) * k - 0.5)
    arcs2 = {q: (v * k if q in ("xc", "a", "b", "b_top", "b_bot", "yc_top", "yc_bot", "y0_shroud") else v)
             for q, v in arcs.items() if not str(q).startswith("_")}
    return cam2, arcs2


def synthesize_strip(p: ImpellerParams, cam: _OffsetCamera, handedness: str, te_phi_deg: float,
                     arcs: dict[str, Any], *, rows: int, blur: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Render *p* through *cam* with a trailing edge of blade 0 at *te_phi_deg* (azimuth from
    the camera direction, + = screen right) and rectify its gap with the same arcs."""
    te_math = -90.0 + te_phi_deg               # the camera sits on -y: the near rim point is at -90 deg
    phase = fastmesh.phase_for_te(p, te_math, handedness)
    V, F, _ = fastmesh.build_mesh(p, handedness=handedness, phase_deg=phase, n_camber=20, n_seg=72)
    img = render3d.render(V, F, cam, blur_sigma=blur, background=0.12)
    strip, _ = rectify_strip(img.astype(np.float32), arcs, rows=rows)
    return strip, img


def _refine_synthesis(res: dict[str, Any], Z: int, beta1: float, debug: dict[str, Any],
                      hands: tuple[str, ...] | None = None) -> dict[str, Any] | None:
    """Sweep beta2 (coarse 2 deg, fine 0.5 deg) per handedness and maximise the similarity of
    the rendered gap strip with the photographed one."""
    arcs, tes, strip = res["_arcs_work"], res["_tes_work"], res["_strip"]
    if tes["model"] is None or len(tes["te_x"]) < 2:
        return None
    t0 = time.perf_counter()
    gray = res["_gray_work"]
    H, W = gray.shape
    k_elev = res["k_elev"]
    gap_px = arcs["yc_bot"] - arcs["yc_top"]
    b2_over_r2 = gap_px / (arcs["a"] * math.sqrt(max(1e-3, 1 - k_elev ** 2)))
    p0 = _synth_params(Z, b2_over_r2, beta1, 40.0)
    size = (W, int(min(H, arcs["yc_bot"] + 0.3 * gap_px)) + 1)
    c_fitted = bool(tes["model"].get("c_fitted"))
    c_init = tes["model"]["c"] if c_fitted else _C_PRIOR
    cam, cam_info = _fit_camera(arcs, p0, size, (W, H), c_init, 0.04 if c_fitted else 0.1)
    if c_fitted and cam_info["rms_px"] > 1.5:          # the TE-derived perspective does not fit the arcs
        cam2, info2 = _fit_camera(arcs, p0, size, (W, H), _C_PRIOR, 0.1)
        if info2["rms_px"] < cam_info["rms_px"]:
            cam, cam_info = cam2, info2
            cam_info["c_from_te_rejected"] = True
    b2_over_r2 = cam_info["b2_over_r2"]
    phis = np.degrees(tes["te_phi"])
    te_phi0 = float(phis[int(np.argmin(np.abs(phis)))])
    # sweeps at reduced resolution (the pattern is coarse); the final comparison at full size
    k = float(np.clip(_SYNTH_SCALE_PX / W, 0.25, 1.0))
    cam_s, arcs_s = _scaled(cam, arcs, k)
    rows_s = max(16, int(round(strip.shape[0] * k)))
    strip_s = cv2.resize(strip, (int(round(strip.shape[1] * k)), rows_s), interpolation=cv2.INTER_AREA)
    f_obs, h_obs = _strip_feature(strip_s)
    if hands is None:
        hands = (res["handedness"],) if res["handedness"] in ("ccw", "cw") else ("ccw", "cw")

    def strip_for(beta2: float, hand: str, te_phi: float) -> np.ndarray:
        p = _synth_params(Z, b2_over_r2, beta1, beta2)
        s_, _ = synthesize_strip(p, cam_s, hand, te_phi, arcs_s, rows=rows_s)
        if s_.shape != strip_s.shape:
            s_ = cv2.resize(s_, (strip_s.shape[1], strip_s.shape[0]), interpolation=cv2.INTER_LINEAR)
        return s_

    def score(beta2: float, hand: str, te_phi: float) -> float:
        f_s, h_s = _strip_feature(strip_for(beta2, hand, te_phi))
        return 0.5 * _ncc(f_obs, f_s) + 0.5 * _ncc(h_obs, h_s)

    # TE phase: the detected TE nearest the centre seeds it; refine by aligning the TE-line
    # pattern (column profile of |gx|) over +- a third of the pitch, for both hands
    pitch = 360.0 / Z
    gx_o, _ = _strip_gradients(strip_s)
    prof_o = gaussian_filter1d(np.abs(gx_o)[int(0.15 * rows_s):int(0.85 * rows_s)].mean(axis=0), 1.5)
    def align_phase(hand: str, betas, centre: float, half: float, n: int = 13) -> float:
        best_ph = None
        for b in betas:
            for dphi in np.linspace(-half, half, n):
                gx_s, _ = _strip_gradients(strip_for(b, hand, centre + dphi))
                prof_s = gaussian_filter1d(np.abs(gx_s)[int(0.15 * rows_s):int(0.85 * rows_s)].mean(axis=0), 1.5)
                v = _ncc(prof_o, prof_s)
                if best_ph is None or v > best_ph[0]:
                    best_ph = (v, float(centre + dphi))
        return best_ph[1]

    te_phi_by_hand = {hand: align_phase(hand, (30.0, 55.0), te_phi0, pitch / 2.0, 15) for hand in hands}
    lo, hi, step = _SYNTH_COARSE
    grid = np.arange(lo, hi + 1e-9, step)
    best = None
    curves = {}
    for hand in hands:
        sc = np.array([score(b, hand, te_phi_by_hand[hand]) for b in grid])
        curves[hand] = sc
        if best is None or sc.max() > best[0]:
            best = (float(sc.max()), hand, sc)
    _, hand, sc = best
    b_c0 = float(grid[int(sc.argmax())])
    te_phi = align_phase(hand, (b_c0,), te_phi_by_hand[hand], pitch / 12.0, 9)
    sc = np.array([score(b, hand, te_phi) for b in grid])        # re-sweep with the aligned phase
    curves[hand] = sc
    b_c = float(grid[int(sc.argmax())])
    fine = np.arange(max(lo, b_c - _SYNTH_FINE_HALF), min(hi, b_c + _SYNTH_FINE_HALF) + 1e-9, _SYNTH_FINE_STEP)
    sf = np.array([score(b, hand, te_phi) for b in fine])
    i = int(sf.argmax())
    b_f = float(fine[i])
    if 0 < i < len(sf) - 1:
        den = sf[i - 1] - 2 * sf[i] + sf[i + 1]
        if den < 0:
            b_f += _SYNTH_FINE_STEP * float(np.clip(0.5 * (sf[i - 1] - sf[i + 1]) / den, -1, 1))
    rng = float(sc.max() - sc.min())
    above = grid[sc >= sc.max() - 0.25 * rng] if rng > 1e-6 else grid
    width = float(above.max() - above.min()) if len(above) else hi - lo
    peak_contrast = float((sc.max() - np.median(sc)) / (sc.std() + 1e-6))
    hand_margin = 0.0
    if len(hands) == 2:
        other = [h for h in hands if h != hand][0]
        hand_margin = float(sc.max() - curves[other].max())
    out = {"beta2": b_f, "handedness": hand, "hand_margin": hand_margin,
           "coarse": {"grid": grid.tolist(), "score": {h: c.tolist() for h, c in curves.items()}},
           "fine": {"grid": fine.tolist(), "score": sf.tolist()}, "ncc": float(sf.max()),
           "peak_width_deg": width, "peak_contrast": peak_contrast, "camera": cam_info,
           "b2_over_r2": b2_over_r2, "te_phi_deg": te_phi, "te_phi_seed_deg": te_phi0, "scale": k,
           "time_s": time.perf_counter() - t0, "n_renders": int(len(grid) * (len(hands) + 1) + len(fine) + 30 * len(hands) + 9)}
    debug["synthesis"] = out
    return out


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------
def analyze_rim_gap(images: list[np.ndarray], *, blade_count: int, outer_diameter_mm: float | None = None,
                    focal_px: float | None = None, beta1_hint: float | None = None,
                    blade_width_mm: float | None = None, refine: bool = True,
                    work_width: int = _WORK_WIDTH) -> RimGapResult:
    """beta2 + handedness from one or more side photos of the rim gap (no marker needed).

    ``outer_diameter_mm``/``focal_px`` (when both are known) fix the camera distance for the
    perspective correction; otherwise it is fitted from the TE spacing (>= 5 TEs) or a far
    camera is assumed and the confidence interval widened.  ``beta1_hint`` is used by the
    synthesis stage (prior 30 deg)."""
    if isinstance(images, np.ndarray):
        images = [images]
    if not images:
        raise ValueError("analyze_rim_gap needs at least one image")
    Z = int(blade_count)
    beta1 = float(beta1_hint) if beta1_hint is not None else _PRIOR_BETA1
    warnings: list[str] = []
    per: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for i, img in enumerate(images):
        c_hint = None
        if outer_diameter_mm and focal_px:
            # c = r2 / D: the rim semi-axis in px ~ f r2 / D -> solved after the arcs; pass later
            c_hint = None
        try:
            r = analyze_rim_gap_image(img, Z, work_width=work_width, c_hint=c_hint)
        except ValueError as exc:
            warnings.append(f"image {i}: {exc}")
            per.append({"index": i, "error": str(exc), "beta2": None, "confidence": 0.0,
                        "handedness": "undetermined", "warnings": [str(exc)]})
            continue
        if outer_diameter_mm and focal_px and r.get("_arcs_work") is not None:
            # A ~ f r2 / D  ->  c = r2 / D = A / f with A ~ a (1 - c): solve c = a (1 - c) / f
            a_px = r["_arcs_work"]["a"] / r["scale"]
            c_val = a_px / (focal_px + a_px)
            try:
                r = analyze_rim_gap_image(img, Z, work_width=work_width, c_hint=float(np.clip(c_val, 0.02, 0.6)))
            except ValueError as exc:                # pragma: no cover - same image, already passed
                warnings.append(f"image {i}: {exc}")
        r["index"] = i
        r["method"] = "sign-change" if r["beta2"] is not None else "none"
        if refine and r.get("_tes_work") is not None:
            try:
                weak = r["beta2"] is None or r["confidence"] < 0.3 or r["handedness"] == "undetermined"
                syn = _refine_synthesis(r, Z, beta1, r["debug"], hands=("ccw", "cw") if weak else None)
            except Exception as exc:    # synthesis must never kill the sign-change answer
                syn = None
                r["warnings"].append(f"synthesis failed: {exc}")
            if syn is not None:
                r["beta2_sign_change"] = r["beta2"]
                r["beta2_synthesis"] = syn["beta2"]
                r["synthesis"] = {k: v for k, v in syn.items() if k not in ("coarse", "fine")}
                # a usable synthesis: a clear peak, or a high similarity at the optimum (broad but
                # strong peaks have a low contrast), or anything decent when the sign-change
                # stage has (next to) no confidence
                clear = (syn["peak_contrast"] >= 1.5 or syn["ncc"] >= 0.45
                         or (syn["ncc"] >= 0.3 and (r["beta2"] is None or r["confidence"] < 0.15)))
                agree = r["beta2"] is None or abs(syn["beta2"] - r["beta2"]) <= max(8.0, 2.0 * (r["sigma_deg"] or 8.0))
                # the synthesis models perspective, curvature and occlusion exactly: it overrides the
                # sign-change estimate unless that one is confident and the two disagree
                if clear and (agree or r["confidence"] < 0.6):
                    if r["beta2"] is not None and not agree:
                        r["warnings"].append(f"synthesis ({syn['beta2']:.1f} deg) overrides a weak sign-change "
                                             f"estimate ({r['beta2']:.1f} deg)")
                    r["beta2"] = syn["beta2"]
                    r["method"] = "synthesis"
                    if r["handedness"] == "undetermined" or r["confidence"] < 0.3:
                        if syn["handedness"] != r["handedness"] and r["handedness"] != "undetermined":
                            r["warnings"].append("synthesis prefers the other handedness")
                        r["handedness"] = syn["handedness"] if (len(syn["coarse"]["score"]) == 1 or syn["hand_margin"] > 0.02) else r["handedness"]
                    r["sigma_deg"] = float(max(0.8, min(syn["peak_width_deg"] / 4.0, r["sigma_deg"] or 99.0)))
                    r["confidence"] = float(np.clip(0.4 * (r["confidence"] or 0.0) + 0.6 * min(1.0, 0.5 + 0.5 * syn["ncc"]), 0, 1))
                elif clear and not agree:
                    r["warnings"].append(f"synthesis ({syn['beta2']:.1f} deg) disagrees with the sign-change "
                                         f"estimate ({r['beta2']:.1f} deg); keeping sign-change")
                    r["confidence"] *= 0.7
        per.append(r)
        for w in r["warnings"]:
            warnings.append(f"image {i}: {w}")

    good = [r for r in per if r.get("beta2") is not None]
    first = per[0] if per else {}
    if not good:
        return RimGapResult(beta2=_PRIOR_BETA2, confidence=0.0, handedness="undetermined", x0_px=None,
                            te_x=[], te_widths=[], rim=None, gap={}, per_image=per, method="prior",
                            debug={"time_s": time.perf_counter() - t_all}, warnings=warnings + ["no usable rim image: prior beta2"])
    w = np.array([1.0 / max(r["sigma_deg"], 2.0) ** 2 for r in good])     # no image is trusted below 2 deg
    b = np.array([r["beta2"] for r in good])
    if len(good) >= 3:
        # robust: images farther than _DISAGREE_DEG from the median count only a tenth
        med = float(np.median(b))
        w = np.where(np.abs(b - med) > _DISAGREE_DEG, 0.1 * w, w)
    beta2 = float(np.sum(w * b) / np.sum(w))
    conf = float(np.sum(w * np.array([r["confidence"] for r in good])) / np.sum(w))
    hands = [r["handedness"] for r in good if r["handedness"] != "undetermined"]
    if hands:
        hand = max(set(hands), key=hands.count)
        if len(set(hands)) > 1:
            warnings.append(f"handedness disagrees between images: {hands}")
            conf *= 0.6
    else:
        hand = "undetermined"
    if len(good) > 1:
        conf *= min(1.0, 0.8 + 0.1 * len(good))
        if float(b.max() - b.min()) > _DISAGREE_DEG:
            warnings.append(f"beta2 disagrees between images by {b.max() - b.min():.1f} deg")
            conf *= 0.6
    methods = {r["method"] for r in good}
    method = "synthesis" if methods == {"synthesis"} or ("synthesis" in methods and len(methods) == 1) else \
        ("synthesis" if all(m == "synthesis" for m in methods) else "sign-change")
    g0 = good[0]
    rim = None
    if g0.get("arcs"):
        a_ = g0["arcs"]
        rim = EllipseFit((a_["xc"], a_["yc_bot"] - a_["b"]), (a_["a"], max(a_["b"], 1e-3)), 0.0, score=g0["arcs"]["support"],
                         meta={"kind": "backplate rim (near arc fit)"})
    return RimGapResult(
        beta2=beta2, confidence=float(np.clip(conf, 0.0, 1.0)), handedness=hand,
        x0_px=g0.get("x0_px"), te_x=list(g0.get("te_x", [])),
        te_widths=[m["width_px"] for m in g0.get("te", [])], rim=rim, gap=dict(g0.get("gap", {})),
        per_image=per, method=method,
        debug={"time_s": time.perf_counter() - t_all, "sigma_deg": float(1.0 / math.sqrt(np.sum(w))),
               "beta1_used": beta1, "first_image": first.get("debug", {})},
        warnings=warnings)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def draw_rim_gap_overlay(img: np.ndarray, result: RimGapResult, index: int = 0) -> np.ndarray:
    """BGR copy of *img* with the rim arcs, the gap strip, the TEs (face side arrows), the
    edge-on transition x0 and a text summary."""
    out = img.copy() if (img.ndim == 3 and img.shape[2] == 3) else cv2.cvtColor(_to_gray(img), cv2.COLOR_GRAY2BGR)
    H, W = out.shape[:2]
    thick = max(1, int(round(W / 700)))
    r = result.per_image[index] if index < len(result.per_image) else None
    if r is None or r.get("beta2") is None and not r.get("arcs"):
        cv2.putText(out, "rim gap: no result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    a_ = r["arcs"]
    xs = np.linspace(a_["xc"] - 0.97 * a_["a"], a_["xc"] + 0.97 * a_["a"], 200)
    for yc, bb_, col in ((a_["yc_top"], a_["b_top"], (0, 200, 255)), (a_["yc_bot"], a_["b_bot"], (0, 255, 0))):
        pts = np.stack([xs, _arc_y(xs, a_["xc"], a_["a"], bb_, yc)], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, col, thick, cv2.LINE_AA)
    y_t, y_b = a_["yc_top"], a_["yc_bot"]

    def arc_t(x):
        return float(_arc_y(np.array([float(x)]), a_["xc"], a_["a"], a_["b_top"], y_t)[0])

    def arc_b(x):
        return float(_arc_y(np.array([float(x)]), a_["xc"], a_["a"], a_["b_bot"], y_b)[0])
    for m in r.get("te", []):
        x = int(round(m["x"]))
        yt, yb = arc_t(m["x"]), arc_b(m["x"])
        cv2.line(out, (x, int(yt)), (x, int(yb)), (255, 128, 0), thick + 1, cv2.LINE_AA)
        if m["side"] != 0:
            ym = int(0.5 * (yt + yb))
            L = int(max(8, 0.04 * a_["a"]))
            cv2.arrowedLine(out, (x, ym), (x + m["side"] * L, ym), (255, 0, 255), thick + 1, cv2.LINE_AA, tipLength=0.4)
        if m.get("psi_deg") is not None:
            cv2.putText(out, f"{m['psi_deg']:+.0f}", (x - 12, int(yb) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 0, 255), 1, cv2.LINE_AA)
    if r.get("x0_px") is not None:
        x0 = int(round(a_["xc"] + r["x0_px"]))
        yt, yb = arc_t(x0), arc_b(x0)
        cv2.line(out, (x0, int(yt) - 10), (x0, int(yb) + 10), (0, 0, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(out, "x0", (x0 + 4, int(yt) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    xc = int(round(a_["xc"]))
    cv2.line(out, (xc, int(y_t) - 20), (xc, int(y_b) + 20), (200, 200, 200), 1, cv2.LINE_AA)
    txt = [f"beta2 {result.beta2:.1f} deg ({result.method}, conf {result.confidence:.2f}), hand {result.handedness}"]
    if r.get("beta2_sign_change") is not None:
        txt.append(f"sign-change {r['beta2_sign_change']:.1f}  synthesis {r.get('beta2_synthesis', float('nan')):.1f}")
    txt.append(f"TEs {r.get('n_te', 0)}, elev ~{math.degrees(math.asin(min(1.0, r.get('k_elev', 0.0)))):.0f} deg, "
               f"c {r.get('c', 0):.2f}{'' if r.get('c_known') else ' (assumed)'}")
    for i, t in enumerate(txt):
        cv2.putText(out, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out
