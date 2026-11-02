"""Recover :class:`ImpellerParams` from an axial-view photograph.

Pipeline (see SPEC "Module: fanscan/analyze.py"):

1. scale        - white reference circle of known diameter (or ``outer_diameter_mm``)
2. centre/disc  - Otsu threshold, largest contour, least-squares circle fit
3. radial       - ``cv2.warpPolar`` -> mean / std vs r -> blade annulus (r1, r2), hub
4. blade count  - harmonic-sum spectrum of angular profiles at several radii, majority vote
5. tracing      - per-radius angular minima linked into Z traces, ``geometry.fit_linear_beta``
6. thickness    - half-depth width of the dark ridge at mid radius, obliquity corrected
7. blade width  - not observable; use the hint or 0.2 * D2 with a warning

Coordinate convention
---------------------
``cv2.warpPolar`` measures its angle phi in the *image* frame (y down):
``phi = atan2(py - cy, px - cx)``.  The blade model (:mod:`fanscan.geometry`)
lives in a y-up math frame where the blade sweeps toward +theta (CCW).  A math
point ``(r, theta)`` is drawn at ``px = cx + r cos(theta)``,
``py = cy - r sin(theta)`` so ``phi = -theta`` and we use ``theta = -phi``.
A wrong sign shows up as ``beta ~ 180 - beta``.
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import geometry
from .params import ImpellerParams

Z_MIN, Z_MAX = 2, 64
_REF_THRESHOLD = 240          # reference circle is the only thing brighter than this
_MIN_ANGLE_BINS = 1440        # polar rows (0.25 deg) - raised so 1 row ~ 1 px at the disc edge
_MAX_ANGLE_BINS = 4096


@dataclass
class AnalysisResult:
    params: ImpellerParams
    px_per_mm: float
    center_xy: tuple[float, float]
    confidence: dict[str, float] = field(default_factory=dict)   # 0..1 per parameter
    debug: dict[str, Any] = field(default_factory=dict)          # intermediate arrays/values


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit of (N,2) points -> (cx, cy, r)."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x * x + y * y)
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -a / 2.0, -bb / 2.0
    r = math.sqrt(max(cx * cx + cy * cy - c, 0.0))
    return float(cx), float(cy), float(r)


def _find_reference_circle(gray: np.ndarray, debug: dict[str, Any]
                           ) -> tuple[float, float, float] | None:
    """Locate the solid white reference circle -> (cx, cy, radius_px) or None.

    Candidate blobs are everything above ``_REF_THRESHOLD`` after a median
    filter.  The most circular blob with a plausible area wins.  Its radius is
    refined from the integral of the normalised intensity over a window around
    the blob (``pi r^2``), which is sub-pixel accurate and blur-invariant.
    """
    h, w = gray.shape
    med = cv2.medianBlur(gray, 5)
    _, bw = cv2.threshold(med, _REF_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_area = float(h * w)
    best, best_score = None, 0.0
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30 or area > 0.2 * img_area:       # too small / this is the impeller
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4.0 * math.pi * area / (per * per)   # 1 for a perfect circle
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * r * r)              # solid disc -> ~1
        score = circ * fill
        candidates.append({"center": (float(cx), float(cy)), "r": float(r),
                           "area": float(area), "circularity": float(circ), "fill": float(fill)})
        if circ > 0.7 and fill > 0.7 and score > best_score:
            best, best_score = (cx, cy, r, area), score
    debug["ref_candidates"] = candidates
    if best is None:
        return None
    cx, cy, r, area = best

    # Sub-pixel radius: integrate (I - bg) / (peak - bg) over a window.
    pad = int(round(r * 1.5 + 8))
    x0, x1 = max(int(cx - pad), 0), min(int(cx + pad) + 1, w)
    y0, y1 = max(int(cy - pad), 0), min(int(cy + pad) + 1, h)
    win = gray[y0:y1, x0:x1].astype(np.float64)
    border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    bg = float(np.median(border))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.7 * r) ** 2
    peak = float(np.median(win[inside])) if inside.any() else 255.0
    if peak - bg < 40:
        r_sub = r
    else:
        # restrict the integral to a disc comfortably larger than the blob so the
        # impeller never leaks into it
        roi = (xx - cx) ** 2 + (yy - cy) ** 2 < (r + 6) ** 2
        s = float(np.sum((win[roi] - bg) / (peak - bg)))
        r_sub = math.sqrt(max(s, 1.0) / math.pi)
        if abs(r_sub - r) > 0.15 * r + 2:     # integral disagrees wildly -> trust the contour
            r_sub = r
    debug["ref_circle"] = {"center": (float(cx), float(cy)), "r_contour": float(r),
                           "r_px": float(r_sub), "bg": bg, "peak": peak}
    return float(cx), float(cy), float(r_sub)


def _find_disc(gray: np.ndarray, ref: tuple[float, float, float] | None,
               debug: dict[str, Any]) -> tuple[float, float, float]:
    """Backplate disc centre and radius from Otsu threshold + circle fit."""
    work = cv2.medianBlur(gray, 5)
    if ref is not None:
        # paint the reference circle with the background level so it never
        # joins the disc contour
        bg = int(np.median(work[:8, :]))
        cv2.circle(work, (int(round(ref[0])), int(round(ref[1]))), int(ref[2] + 6), bg, -1)
    thr, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no backplate disc found (empty threshold image)")
    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(float)
    (mx, my), mr = cv2.minEnclosingCircle(c)
    cx, cy, r = _fit_circle(pts)
    # Robust re-fit: drop contour points far from the first fit (ref circle
    # fragments, noise blobs touching the rim).
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    keep = np.abs(d - r) < max(3.0, 0.03 * r)
    if keep.sum() > 20:
        cx, cy, r = _fit_circle(pts[keep])
    debug["disc"] = {"otsu_threshold": float(thr), "center": (cx, cy), "r_px": r,
                     "min_enclosing": ((float(mx), float(my)), float(mr)),
                     "contour_points": int(len(pts))}
    if r < 10:
        raise ValueError("backplate disc too small to analyse")
    return cx, cy, r


def _polar(gray: np.ndarray, cx: float, cy: float, max_r: float
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Polar unwrap: rows = angle (image frame, 0..2pi), cols = radius in px."""
    R = int(round(max_r))
    H = int(np.clip(round(2.0 * math.pi * max_r), _MIN_ANGLE_BINS, _MAX_ANGLE_BINS))
    H += H % 4
    pol = cv2.warpPolar(gray, (R, H), (float(cx), float(cy)), float(R),
                        cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR)
    r_px = np.arange(R, dtype=float) * (float(R) / R)
    return pol.astype(np.float32), r_px, 2.0 * math.pi / H


def _edge_cross(prof: np.ndarray, i0: int, level: float, direction: int) -> float:
    """Walk from index i0 in *direction* until *prof* crosses *level*; sub-sample."""
    n = len(prof)
    i = i0
    above = prof[i] > level
    while 0 <= i + direction < n:
        j = i + direction
        if (prof[j] > level) != above:
            a, b = prof[i], prof[j]
            t = (level - a) / (b - a) if b != a else 0.5
            return float(i + direction * t)
        i = j
    return float(i)


def _blade_band(std_prof: np.ndarray, disc_r: float, debug: dict[str, Any]
                ) -> tuple[float, float, float]:
    """Inner/outer edge of the high-angular-variance annulus -> (r1, r2, contrast).

    Works on the *variance* (std^2): for a two-level row the variance is linear
    in the blade coverage fraction, so the half-level crossing sits at half
    coverage, i.e. at the blade end itself.  (std ~ sqrt(coverage) would bias
    the edge inward/outward by a fraction of the blade thickness.)
    """
    s = gaussian_filter1d(std_prof.astype(np.float64) ** 2, 1.5)
    lim = int(min(disc_r, len(s) - 1))
    valid = s[3:lim]
    floor = float(np.percentile(valid, 20))
    peak = float(np.percentile(valid, 90))
    if peak - floor < 16.0:      # std contrast < 4 grey levels
        raise ValueError("no blade pattern found (angular variance is flat)")
    thr = floor + 0.4 * (peak - floor)
    above = s > thr
    above[:3] = False
    above[lim:] = False
    # longest contiguous run above threshold
    runs, start = [], None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))
    if not runs:
        raise ValueError("no blade annulus found")
    a, b = max(runs, key=lambda ab: ab[1] - ab[0])
    width = b - a
    if width < 8:
        raise ValueError("blade annulus too narrow")
    # refine each edge at the half level between the floor and the local band level
    inner_lvl = float(np.median(s[a + max(2, width // 10): a + max(4, width // 4)]))
    outer_lvl = float(np.median(s[b - max(4, width // 4): b - max(2, width // 10) + 1]))
    r1 = _edge_cross(s, a + max(2, width // 10), 0.5 * (floor + inner_lvl), -1)
    r2 = _edge_cross(s, b - max(2, width // 10), 0.5 * (floor + outer_lvl), +1)
    contrast = math.sqrt(max(0.5 * (inner_lvl + outer_lvl) - floor, 0.0)) / math.sqrt(max(floor, 1.0))
    debug["band"] = {"floor_var": floor, "peak_var": peak, "threshold_var": thr,
                     "coarse": (int(a), int(b)),
                     "r1_px": r1, "r2_px": r2, "inner_level": inner_lvl,
                     "outer_level": outer_lvl}
    return r1, r2, float(contrast)


def _hub_radius(mean_prof: np.ndarray, r1: float, debug: dict[str, Any]
                ) -> tuple[float | None, float]:
    """Hub radius from the step in the mean profile below r1 -> (r_hub_px, confidence)."""
    m = gaussian_filter1d(mean_prof, 1.5)
    lo, hi = 3, int(r1) - 3
    if hi - lo < 8:
        return None, 0.0
    seg = m[lo:hi]
    g = np.gradient(seg)
    k = int(np.argmax(np.abs(g))) + lo
    half = max(3, int(0.08 * r1))
    lvl_in = float(np.median(m[max(lo, k - 3 * half): max(lo + 1, k - half)]))
    lvl_out = float(np.median(m[min(hi - 1, k + half): min(hi, k + 3 * half) + 1]))
    step = lvl_out - lvl_in
    debug["hub"] = {"step_index": k, "level_in": lvl_in, "level_out": lvl_out}
    if abs(step) < 12:
        return None, 0.0
    r_hub = _edge_cross(m, max(lo, k - 2 * half), 0.5 * (lvl_in + lvl_out), +1)
    debug["hub"]["r_hub_px"] = r_hub
    conf = float(np.clip(abs(step) / 60.0, 0.0, 1.0))
    return r_hub, conf


def _blade_count(pol: np.ndarray, c_lo: int, c_hi: int, hint: int | None,
                 debug: dict[str, Any]) -> tuple[int, float]:
    """Majority vote of the harmonic-sum spectrum over radii -> (Z, agreement)."""
    H = pol.shape[0]
    cols = np.unique(np.linspace(c_lo, c_hi, 24).round().astype(int))
    votes: dict[int, int] = {}
    kmax = min(Z_MAX, H // 2 - 1)
    for c in cols:
        row = pol[:, c].astype(np.float64)
        row -= row.mean()
        mag = np.abs(np.fft.rfft(row))
        score = np.zeros(kmax + 1)
        for k in range(Z_MIN, kmax + 1):
            # sum of the first harmonics: the fundamental beats its multiples,
            # and sub-multiples only collect noise bins
            tot, n = 0.0, 0
            for mlt in (1, 2, 3):
                if mlt * k < len(mag):
                    tot += mag[mlt * k]
                    n += 1
            score[k] = tot / max(n, 1) + 0.5 * mag[k]
        k = int(np.argmax(score))
        votes[k] = votes.get(k, 0) + 1
    debug["blade_count_votes"] = votes
    best = max(votes.values())
    tied = sorted(k for k, v in votes.items() if v == best)
    if hint is not None and hint in tied:
        Z = hint
    else:
        Z = tied[0]
    return Z, best / float(len(cols))


def _ridge_center_width(row: np.ndarray, m: int, H: int, base: float | None = None,
                        max_half: int | None = None) -> tuple[float, float] | None:
    """Half-depth centre and width (rows) of the dark ridge around row index m.

    The centre is the midpoint of the two half-depth crossings, which is far
    more stable than the arg-min for a flat-bottomed (unblurred) trough in a
    noisy row.  Returns None when the ridge is too shallow or the crossings
    are not found within *max_half* rows.
    """
    if base is None:
        base = float(np.median(row))
    mi = int(round(m)) % H
    depth = base - row[mi]
    if depth < 8:
        return None
    level = row[mi] + 0.5 * depth
    limit = H // 2 if max_half is None else max(2, int(max_half))

    def walk(direction: int) -> float:
        i = mi
        for k in range(limit):
            j = (i + direction) % H
            if row[j] >= level:
                a, b = row[i], row[j]
                t = (level - a) / (b - a) if b != a else 0.5
                return k + t
            i = j
        return float("nan")

    wl, wr = walk(-1), walk(+1)
    if not (np.isfinite(wl) and np.isfinite(wr)):
        return None
    return mi + 0.5 * (wr - wl), float(wl + wr)


def _row_minima(row: np.ndarray, Z: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel positions (rows, circular) and prominences of the Z deepest minima."""
    pad = H // 4
    ext = np.concatenate([row[-pad:], row, row[:pad]])
    spread = float(np.percentile(row, 95) - np.percentile(row, 5))
    peaks, props = find_peaks(-ext, distance=max(2, int(0.45 * H / Z)),
                              prominence=0.25 * max(spread, 1.0))
    pos = peaks - pad
    ok = (pos >= 0) & (pos < H)
    pos, prom = pos[ok], props["prominences"][ok]
    if len(pos) == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(prom)[::-1][:Z]
    pos, prom = pos[order], prom[order]
    base = float(np.median(row))
    refined = []
    for p in pos:
        cw = _ridge_center_width(row, int(p), H, base=base, max_half=int(0.45 * H / Z))
        if cw is not None:
            refined.append(cw[0])
            continue
        # fallback: parabolic refinement of the minimum
        y0, y1, y2 = ext[p + pad - 1], ext[p + pad], ext[p + pad + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        refined.append(p + float(np.clip(off, -1, 1)))
    return np.asarray(refined), prom


def _fit_blade(r_mm: np.ndarray, theta: np.ndarray, r1_mm: float, r2_mm: float
               ) -> tuple[float, float, float]:
    """(beta1, beta2, rms_deg): linear-beta model fitted to one trace.

    ``geometry.fit_linear_beta`` gives the initial guess from pointwise
    gradients; it is then refined by least squares of the *integrated* camber
    line against the trace, which removes the upward bias that gradient noise
    induces through atan(1/x).
    """
    from scipy.optimize import least_squares

    b1, b2 = geometry.fit_linear_beta(r_mm, theta, r1_mm, r2_mm)
    b1, b2 = float(np.clip(b1, 2.0, 178.0)), float(np.clip(b2, 2.0, 178.0))

    def model(x: np.ndarray) -> np.ndarray:
        rr, th = geometry.camber_line(r1_mm, r2_mm, x[0], x[1], n=96)
        return np.interp(r_mm, rr, th) + x[2]

    def resid(x: np.ndarray) -> np.ndarray:
        return model(x) - theta

    x0 = np.array([b1, b2, float(np.mean(theta - model(np.array([b1, b2, 0.0]))))])
    sol = least_squares(resid, x0, bounds=([1.0, 1.0, -np.inf], [179.0, 179.0, np.inf]),
                        x_scale=[10.0, 10.0, 0.1], max_nfev=200)
    rms = float(np.degrees(np.sqrt(np.mean(sol.fun ** 2))))
    return float(sol.x[0]), float(sol.x[1]), rms


def _trace_blades(pol: np.ndarray, Z: int, c_lo: int, c_hi: int, dphi: float,
                  debug: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Link per-radius minima into Z traces of (col, row_unwrapped)."""
    H = pol.shape[0]
    sigma = float(np.clip(0.02 * H / Z, 1.0, 5.0))
    sm = gaussian_filter1d(pol, sigma, axis=0, mode="wrap")
    cols = np.arange(c_lo, c_hi + 1)
    minima = {int(c): _row_minima(sm[:, c], Z, H) for c in cols}

    # seed at the column nearest to mid radius that yields exactly Z minima
    mid = (c_lo + c_hi) // 2
    seed = None
    for d in range(0, (c_hi - c_lo) // 2 + 1):
        for c in (mid - d, mid + d):
            if c_lo <= c <= c_hi and len(minima[c][0]) == Z:
                seed = c
                break
        if seed is not None:
            break
    if seed is None:
        raise ValueError(f"could not find {Z} blade minima at any radius")
    start = np.sort(minima[seed][0])
    trace = np.full((Z, len(cols)), np.nan)
    trace[:, seed - c_lo] = start
    tol = 0.35 * H / Z

    def step(c_from: int, c_to: int) -> None:
        cur = trace[:, c_from - c_lo].copy()
        # carry the last known angle for blades with gaps
        for i in range(Z):
            if np.isnan(cur[i]):
                prev = trace[i, :]
                good = np.where(~np.isnan(prev))[0]
                cur[i] = prev[good[np.argmin(np.abs(good - (c_from - c_lo)))]]
        cand, _ = minima[c_to]
        used = np.zeros(len(cand), bool)
        # greedy nearest-angle matching, best matches first
        pairs = []
        for i in range(Z):
            for j, a in enumerate(cand):
                d = (a - cur[i] + H / 2) % H - H / 2       # signed circular diff in rows
                pairs.append((abs(d), i, j, d))
        pairs.sort()
        done = np.zeros(Z, bool)
        for ad, i, j, d in pairs:
            if ad > tol or done[i] or used[j]:
                continue
            trace[i, c_to - c_lo] = cur[i] + d
            done[i] = used[j] = True

    for c in range(seed + 1, c_hi + 1):
        step(c - 1, c)
    for c in range(seed - 1, c_lo - 1, -1):
        step(c + 1, c)

    out = []
    for i in range(Z):
        rows = trace[i]
        ok = ~np.isnan(rows)
        out.append({"col": cols[ok].astype(float), "row": rows[ok],
                    "coverage": float(ok.mean())})
    debug["trace_sigma_rows"] = sigma
    debug["trace_seed_col"] = seed
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def analyze(img: np.ndarray, *, ref_diameter_mm: float | None = None,
            outer_diameter_mm: float | None = None,
            blade_width_mm: float | None = None,
            blade_count_hint: int | None = None) -> AnalysisResult:
    """Analyse an axial impeller photo.  See module docstring for the pipeline."""
    if ref_diameter_mm is None and outer_diameter_mm is None:
        raise ValueError("need ref_diameter_mm or outer_diameter_mm to establish scale")
    gray = _to_gray(np.asarray(img))
    debug: dict[str, Any] = {}
    warnings: list[str] = []
    conf: dict[str, float] = {}

    # 1. scale reference ----------------------------------------------------
    ref = None
    px_per_mm: float | None = None
    if ref_diameter_mm is not None:
        ref = _find_reference_circle(gray, debug)
        if ref is None:
            if outer_diameter_mm is None:
                raise ValueError("reference circle not found and no outer_diameter_mm given")
            warnings.append("reference circle not found; scale taken from outer_diameter_mm")
        else:
            px_per_mm = 2.0 * ref[2] / float(ref_diameter_mm)

    # 2. centre + backplate disc -------------------------------------------
    cx, cy, disc_r = _find_disc(gray, ref, debug)

    # 3. polar unwrap, radial profiles ------------------------------------
    pol, r_px, dphi = _polar(gray, cx, cy, disc_r * 1.06)
    mean_prof = pol.mean(axis=0)
    std_prof = pol.std(axis=0)
    r1_px, r2_px, contrast = _blade_band(std_prof, disc_r, debug)
    r_hub_px, hub_conf = _hub_radius(mean_prof, r1_px, debug)
    debug.update({"polar": pol, "r_px": r_px, "dphi": dphi, "mean_profile": mean_prof,
                  "std_profile": std_prof})

    if px_per_mm is None:
        px_per_mm = 2.0 * r2_px / float(outer_diameter_mm)   # type: ignore[arg-type]
        conf["scale"] = 0.7
    else:
        conf["scale"] = 0.95

    # 4. blade count --------------------------------------------------------
    w_px = r2_px - r1_px
    c_lo = int(math.ceil(r1_px + 0.15 * w_px))
    c_hi = int(math.floor(r2_px - 0.15 * w_px))
    c_hi = min(c_hi, pol.shape[1] - 1)
    if c_hi - c_lo < 6:
        raise ValueError("blade annulus too narrow for analysis")
    Z, agreement = _blade_count(pol, c_lo, c_hi, blade_count_hint, debug)
    if blade_count_hint is not None and Z != blade_count_hint:
        warnings.append(f"blade_count_hint={blade_count_hint} disagrees with detected Z={Z}")

    # 5. blade tracing ------------------------------------------------------
    t_lo = int(math.ceil(r1_px + 0.08 * w_px))
    t_hi = int(math.floor(r2_px - 0.08 * w_px))
    t_hi = min(t_hi, pol.shape[1] - 1)
    traces = _trace_blades(pol, Z, t_lo, t_hi, dphi, debug)
    r1_mm, r2_mm = r1_px / px_per_mm, r2_px / px_per_mm
    betas: list[tuple[float, float]] = []
    trace_xy: list[dict[str, np.ndarray]] = []
    fit_rms: list[float] = []
    for tr in traces:
        if tr["coverage"] < 0.5 or len(tr["col"]) < 8:
            continue
        r = tr["col"] / px_per_mm                     # mm (1 column = 1 px)
        theta = -tr["row"] * dphi                     # image angle -> math frame (y-up, CCW)
        theta = theta - theta[0]
        try:
            b1, b2, rms = _fit_blade(r, theta, r1_mm, r2_mm)
        except ValueError:
            continue
        fit_rms.append(rms)
        betas.append((b1, b2))
        trace_xy.append({"r_px": tr["col"], "theta_math": theta + 0.0,
                         "theta_img_rows": tr["row"], "beta": (b1, b2)})
    if len(betas) < max(1, Z // 3):
        raise ValueError(f"only {len(betas)} of {Z} blades could be traced")
    b_arr = np.asarray(betas)
    beta1 = float(np.median(b_arr[:, 0]))
    beta2 = float(np.median(b_arr[:, 1]))
    if beta1 > 95.0 or beta2 > 95.0:
        warnings.append("blade angle > 95 deg detected (forward-curved, or a theta sign "
                        "convention mismatch: check for beta ~ 180 - expected)")
    beta_spread = float(np.median(np.abs(b_arr - np.median(b_arr, axis=0)).max(axis=1)))
    beta_conf = float(np.clip(1.0 - beta_spread / 10.0, 0.0, 1.0)
                      * np.clip(len(betas) / Z, 0.0, 1.0)
                      * np.clip(1.0 - np.median(fit_rms) / 5.0, 0.2, 1.0))

    # 6. blade thickness ----------------------------------------------------
    r_mid_px = 0.5 * (r1_px + r2_px)
    mid_cols = [int(round(r_mid_px + d)) for d in (-6, -3, 0, 3, 6)]
    H = pol.shape[0]
    light = gaussian_filter1d(pol, 1.0, axis=0, mode="wrap")
    widths_px: list[float] = []
    for tr in trace_xy:
        b1, b2 = tr["beta"]
        for c in mid_cols:
            if c < t_lo or c > t_hi:
                continue
            rows_at = np.interp(c, tr["r_px"], tr["theta_img_rows"])
            # start the walk from the actual minimum near the trace, so a flat
            # noisy trough does not stop the half-depth search early
            win = np.arange(int(round(rows_at)) - 3, int(round(rows_at)) + 4) % H
            m0 = int(win[np.argmin(light[win, c])])
            cw = _ridge_center_width(light[:, c], m0, H, max_half=int(0.45 * H / Z))
            if cw is None:
                continue
            w_rows = cw[1]
            frac = (c / px_per_mm - r1_mm) / max(r2_mm - r1_mm, 1e-6)
            beta_c = math.radians(b1 + (b2 - b1) * frac)
            w_ang_px = w_rows * dphi * c                     # chord of the ridge along the circle
            widths_px.append(w_ang_px * abs(math.sin(beta_c)))
    if widths_px:
        thickness_mm = float(np.median(widths_px)) / px_per_mm
        t_conf = float(np.clip(1.0 - np.std(widths_px) / max(np.median(widths_px), 1e-6), 0.1, 0.8))
    else:
        thickness_mm = 0.02 * (2.0 * r2_mm)
        t_conf = 0.1
        warnings.append("blade thickness not measurable; using 2 % of D2")
    # thickness must satisfy params.validate(): t*4 <= r2 - r1
    thickness_mm = float(min(thickness_mm, 0.24 * (r2_mm - r1_mm)))
    thickness_mm = max(thickness_mm, 0.05)

    # 7. blade width (axial) -----------------------------------------------
    d2_mm = 2.0 * r2_mm
    if blade_width_mm is not None:
        blade_width = float(blade_width_mm)
        conf["blade_width"] = 1.0
    else:
        blade_width = 0.2 * d2_mm
        conf["blade_width"] = 0.1
        warnings.append("blade_width (axial) is not observable in an axial photo; "
                        "estimated as 0.2 * D2")

    # hub fallback ----------------------------------------------------------
    d1_mm = 2.0 * r1_mm
    if r_hub_px is None:
        hub_mm = 0.33 * d1_mm
        warnings.append("hub step not found in the radial profile; estimated as 0.33 * D1")
        hub_conf = 0.1
    else:
        hub_mm = 2.0 * r_hub_px / px_per_mm
        if not (0 < hub_mm < d1_mm):
            hub_mm = 0.33 * d1_mm
            warnings.append("hub estimate implausible; estimated as 0.33 * D1")
            hub_conf = 0.1

    conf.update({
        "outer_diameter": float(np.clip(0.5 + 0.1 * contrast, 0.3, 0.95)),
        "inner_diameter": float(np.clip(0.5 + 0.1 * contrast, 0.3, 0.95)),
        "hub_diameter": hub_conf,
        "blade_count": float(agreement),
        "blade_angle_inlet": beta_conf,
        "blade_angle_outlet": beta_conf,
        "blade_thickness": t_conf,
    })

    params = ImpellerParams(
        outer_diameter=round(d2_mm, 3),
        inner_diameter=round(d1_mm, 3),
        hub_diameter=round(hub_mm, 3),
        blade_count=int(Z),
        blade_angle_inlet=round(float(np.clip(beta1, 1.0, 179.0)), 2),
        blade_angle_outlet=round(float(np.clip(beta2, 1.0, 179.0)), 2),
        blade_width=round(blade_width, 3),
        blade_thickness=round(thickness_mm, 3),
        meta={
            "source": "fanscan.analyze",
            "px_per_mm": float(px_per_mm),
            "center_xy": [float(cx), float(cy)],
            "warnings": warnings,
            "per_blade_beta": [[round(float(a), 2), round(float(b), 2)] for a, b in betas],
            "blade_count_votes": {str(k): int(v) for k, v in debug["blade_count_votes"].items()},
            "confidence": dict(conf),
        },
    )
    debug.update({
        "ref": ref, "disc_r_px": disc_r, "r1_px": r1_px, "r2_px": r2_px,
        "r_hub_px": r_hub_px, "traces": trace_xy, "fit_rms_deg": fit_rms,
        "thickness_samples_px": widths_px, "Z": Z,
    })
    return AnalysisResult(params=params, px_per_mm=float(px_per_mm),
                          center_xy=(float(cx), float(cy)), confidence=conf, debug=debug)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def draw_overlay(img: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """BGR copy of *img* with the detected circles and blade traces drawn."""
    gray = _to_gray(np.asarray(img))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cx, cy = result.center_xy
    s = result.px_per_mm
    p = result.params
    d = result.debug
    c_int = (int(round(cx)), int(round(cy)))

    def circ(r_px: float, color: tuple[int, int, int], th: int = 1) -> None:
        cv2.circle(out, c_int, int(round(r_px)), color, th, cv2.LINE_AA)

    circ(d.get("disc_r_px", p.r2 * s), (0, 200, 255), 1)       # backplate disc
    circ(p.r2 * s, (0, 0, 255), 2)                               # blade tip
    circ(p.r1 * s, (255, 0, 0), 2)                               # leading edge
    circ(p.r_hub * s, (255, 0, 255), 2)                          # hub
    cv2.drawMarker(out, c_int, (0, 255, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
    if d.get("ref") is not None:
        rx, ry, rr = d["ref"]
        cv2.circle(out, (int(round(rx)), int(round(ry))), int(round(rr)), (0, 255, 0), 2,
                   cv2.LINE_AA)

    # traced centres (green dots) + fitted model (yellow)
    for tr in d.get("traces", []):
        r = tr["r_px"]
        th = -tr["theta_img_rows"] * d["dphi"]            # math frame angle (absolute)
        x, y = r * np.cos(th), r * np.sin(th)
        pts = np.stack([cx + x, cy - y], axis=1).round().astype(np.int32)
        for x_, y_ in pts[::3]:
            cv2.circle(out, (int(x_), int(y_)), 1, (0, 255, 0), -1)
        b1, b2 = tr["beta"]
        rr_mm, th_m = geometry.camber_line(p.r1, p.r2, b1, b2, n=96)
        # anchor the model at the trace's first point
        th0 = float(th[0]) - float(np.interp(r[0] / s, rr_mm, th_m))
        xm, ym = geometry.polar_to_xy(rr_mm * s, th_m, th0)
        poly = np.stack([cx + xm, cy - ym], axis=1).round().astype(np.int32)
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)

    label = (f"Z={p.blade_count} D2={p.outer_diameter:.1f} D1={p.inner_diameter:.1f} "
             f"hub={p.hub_diameter:.1f} b1={p.blade_angle_inlet:.1f} "
             f"b2={p.blade_angle_outlet:.1f} t={p.blade_thickness:.2f}")
    cv2.putText(out, label, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m fanscan.analyze",
                                 description="Analyse an axial impeller photo -> params JSON.")
    ap.add_argument("image", help="input PNG/JPG")
    ap.add_argument("--out", help="write ImpellerParams JSON here")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-diameter-mm", type=float)
    g.add_argument("--outer-diameter-mm", type=float)
    ap.add_argument("--blade-width-mm", type=float)
    ap.add_argument("--blade-count-hint", type=int)
    ap.add_argument("--overlay", help="overlay PNG path (default: <image>.overlay.png)")
    a = ap.parse_args(argv)

    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"could not read {a.image}", file=sys.stderr)
        return 2
    res = analyze(img, ref_diameter_mm=a.ref_diameter_mm, outer_diameter_mm=a.outer_diameter_mm,
                  blade_width_mm=a.blade_width_mm, blade_count_hint=a.blade_count_hint)
    out = res.params.to_dict()
    out["confidence"] = res.confidence
    out["px_per_mm"] = res.px_per_mm
    out["center_xy"] = list(res.center_xy)
    print(json.dumps(out, indent=2))
    if a.out:
        res.params.save(a.out)
    ov_path = Path(a.overlay) if a.overlay else Path(a.image).with_suffix(".overlay.png")
    cv2.imwrite(str(ov_path), draw_overlay(img, res))
    print(f"wrote overlay {ov_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
