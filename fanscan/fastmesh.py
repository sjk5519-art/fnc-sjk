"""Cheap numpy-only impeller mesh for synthesis (fanscan v0.3, SPEC3).

``cad.build_impeller`` needs CadQuery and ~10-15 s per model; the rim-gap synthesis stage
(:mod:`fanscan.rimgap`) renders dozens of candidate impellers, so it needs a mesh in well under
50 ms.  This module builds the same geometry *approximately* with plain numpy:

* backplate: annulus r in [r_bore, r2 + BACKPLATE_RIM_MM] (disc when there is no bore), z in [0, tb]
* hub: cylinder r_hub from z = tb to the hub top (``hub_height`` or b1 + tb), annular top
* blades: ``geometry.camber_line`` offset by t/2 (``geometry.offset_camber``), bottom at z = tb,
  top following the shroud underside under the shroud and the linear taper inside the eye
  (same formulas as cad.blade_top_z); side walls + top/bottom caps as triangle strips
* shroud (v0.2 cone + inlet ring): the closed ``cad.shroud_profile`` polygon revolved as quad
  rings; a flat ring r1..r2 for the v0.1 shroud; nothing for open impellers

Nothing is unioned - the parts simply overlap, which is all a z-buffer renderer needs.  Faces
are two-sided in render3d so winding does not matter.

Conventions match cad.py / render3d.py: z up, origin at the backplate bottom centre, mm.
``handedness='ccw'`` sweeps the blade toward +theta as r grows (seen from +z = the inlet),
``'cw'`` mirrors it (theta -> -theta).  ``phase_deg`` rotates the whole blade set; use
:func:`phase_for_te` to put a trailing edge at a given azimuth.

Public API
----------
    build_mesh(params, *, handedness='ccw', phase_deg=0.0, n_camber=24, n_seg=96,
               with_hub=True, with_shroud=True) -> (V (n,3) float64, F (m,3) int64, ids (m,) int32)
    phase_for_te(params, te_azimuth_deg, handedness='ccw') -> float
    wrap_deg(params) -> float
"""
from __future__ import annotations

import math


import numpy as np

from . import geometry
from .params import ImpellerParams

__all__ = ["build_mesh", "phase_for_te", "wrap_deg", "PART_IDS", "BACKPLATE_RIM_MM"]

BACKPLATE_RIM_MM = 2.0          # must match cad.BACKPLATE_RIM_MM
PART_IDS = {"backplate": 0, "hub": 1, "shroud": 2, "blade": 3, "marker": 4}   # = render3d.PART_IDS
_OVERLAP = 0.3                  # blade top pushed this far into the shroud (cad.SHROUD_OVERLAP_MM)
_ROOT_OVERLAP = 0.5             # blade root this far inside the backplate (cad.ROOT_OVERLAP_MM)


# ---------------------------------------------------------------------------
# analytic profile helpers (copies of cad.py formulas; cad imports cadquery, which is slow)
# ---------------------------------------------------------------------------
def _is_shrouded_v2(p: ImpellerParams) -> bool:
    return bool(p.shroud) and p.eye_diameter is not None


def _hub_top(p: ImpellerParams) -> float:
    if p.hub_height is not None:
        return float(p.hub_height)
    b = p.b1 if _is_shrouded_v2(p) else p.blade_width
    return float(b + p.backplate_thickness)


def _blade_height(p: ImpellerParams, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    if not _is_shrouded_v2(p):
        return np.full_like(r, float(p.blade_width))
    t = np.clip((p.r2 - r) / max(p.r2 - p.r1, 1e-9), 0.0, 1.0)
    return p.b2 + (p.b1 - p.b2) * t


def _shroud_underside(p: ImpellerParams, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    tb = float(p.backplate_thickness)
    if not _is_shrouded_v2(p):
        return np.full_like(r, tb + float(p.blade_width))
    frac = np.clip((p.r2 - r) / max(p.r2 - p.r_eye, 1e-9), 0.0, 1.0)
    return tb + p.b2 + float(p.shroud_cone_height) * frac


def _blade_top(p: ImpellerParams, r: np.ndarray) -> np.ndarray:
    """Finished blade top edge z(r) (cad.blade_top_z)."""
    r = np.asarray(r, float)
    tb = float(p.backplate_thickness)
    taper = tb + _blade_height(p, r)
    if not p.shroud:
        return taper
    eps = min(_OVERLAP, float(p.shroud_thickness) / 3.0)
    under = _shroud_underside(p, r) + eps
    if not _is_shrouded_v2(p):
        return np.minimum(taper, under)
    return np.where(r >= p.r_eye, under, taper)


def _shroud_profile(p: ImpellerParams) -> list[tuple[float, float]]:
    """Closed (r, z) polygon of the v0.2 shroud (cad.shroud_profile)."""
    tb, ts = float(p.backplate_thickness), float(p.shroud_thickness)
    r2, r_eye = float(p.r2), float(p.r_eye)
    cone, ring_h = float(p.shroud_cone_height), float(p.shroud_inlet_height)
    z_rim = tb + p.b2
    z_ct = z_rim + cone
    span = max(r2 - r_eye, 1e-6)
    r_in = r_eye + ts
    z_up_at_r_in = z_ct - cone * ts / span + ts
    pts: list[tuple[float, float]] = [(r_eye, z_ct), (r2, z_rim), (r2, z_rim + ts)]
    if ring_h > ts + 1e-6 and r_in < r2:
        pts += [(r_in, z_up_at_r_in), (r_in, z_ct + ring_h), (r_eye, z_ct + ring_h)]
    else:
        pts += [(r_eye, z_ct + ts)]
    return pts


def wrap_deg(p: ImpellerParams) -> float:
    """Blade wrap angle r1 -> r2 [deg] under the linear-beta model."""
    return geometry.wrap_from_betas(p.r1, p.r2, p.blade_angle_inlet, p.blade_angle_outlet)


def phase_for_te(p: ImpellerParams, te_azimuth_deg: float, handedness: str = "ccw") -> float:
    """``phase_deg`` that puts the trailing edge of blade 0 at *te_azimuth_deg* (math frame,
    +x = 0, CCW seen from +z)."""
    w = wrap_deg(p)
    return float(te_azimuth_deg - w) if handedness == "ccw" else float(te_azimuth_deg + w)


# ---------------------------------------------------------------------------
# primitive builders (each returns (V, F) with F referring to its own V)
# ---------------------------------------------------------------------------
def _ring_quads(n: int, offset_a: int, offset_b: int, closed: bool = True) -> np.ndarray:
    """Triangles joining ring a (vertices offset_a..offset_a+n-1) to ring b."""
    i = np.arange(n)
    j = (i + 1) % n if closed else np.minimum(i + 1, n - 1)
    if not closed:
        i, j = i[:-1], j[:-1]
    a0, a1, b0, b1 = offset_a + i, offset_a + j, offset_b + i, offset_b + j
    return np.concatenate([np.stack([a0, a1, b1], 1), np.stack([a0, b1, b0], 1)], 0)


def _subdivide(profile: list[tuple[float, float]], max_len: float, closed: bool) -> list[tuple[float, float]]:
    """Insert points so that no profile edge is longer than *max_len* (keeps the revolved
    triangles small: the z-buffer rasteriser's cost grows with the triangle bounding boxes)."""
    pts = [tuple(map(float, q)) for q in profile]
    out: list[tuple[float, float]] = []
    m = len(pts)
    for k in range(m if closed else m - 1):
        a, b = pts[k], pts[(k + 1) % m]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(L / max_len)))
        for i in range(n):
            f = i / n
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    if not closed:
        out.append(pts[-1])
    return out


def _revolve(profile: list[tuple[float, float]], n_seg: int, closed: bool = True,
             max_len: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a polyline of (r, z) points about z: one vertex ring per profile point."""
    if max_len is not None:
        profile = _subdivide(profile, max_len, closed)
        closed = closed and True
    t = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    rings = []
    for r, z in profile:
        rings.append(np.stack([r * ct, r * st, np.full(n_seg, float(z))], 1))
    V = np.concatenate(rings, 0)
    faces = []
    m = len(profile)
    for k in range(m if closed else m - 1):
        a, b = k, (k + 1) % m
        ra, rb = profile[a][0], profile[b][0]
        if abs(ra) < 1e-9 and abs(rb) < 1e-9:
            continue
        faces.append(_ring_quads(n_seg, a * n_seg, b * n_seg))
    F = np.concatenate(faces, 0) if faces else np.zeros((0, 3), np.int64)
    return V, F


def _fan(radius: float, z: float, n_seg: int, r_in: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat disc (r_in = 0) or annulus at height z."""
    step = radius / 4.0
    if r_in > 1e-9:
        return _revolve([(r_in, z), (radius, z)], n_seg, closed=False, max_len=step)
    # centre fan up to radius / 4, then rings
    r1 = radius / 4.0
    t = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    V = np.concatenate([[[0.0, 0.0, z]], np.stack([r1 * np.cos(t), r1 * np.sin(t), np.full(n_seg, z)], 1)], 0)
    i = np.arange(n_seg) + 1
    j = (np.arange(n_seg) + 1) % n_seg + 1
    F = np.stack([np.zeros(n_seg, np.int64), i, j], 1)
    V2, F2 = _revolve([(r1, z), (radius, z)], n_seg, closed=False, max_len=step)
    return np.concatenate([V, V2], 0), np.concatenate([F, F2 + len(V)], 0)


def _blade(p: ImpellerParams, theta0: float, sign: float, n_camber: int
           ) -> tuple[np.ndarray, np.ndarray]:
    r, th = geometry.camber_line(p.r1, p.r2, p.blade_angle_inlet, p.blade_angle_outlet, n=n_camber)
    poly = geometry.offset_camber(r, sign * th, float(p.blade_thickness), theta0)   # (2n, 2)
    n = len(r)
    rr = np.hypot(poly[:, 0], poly[:, 1])
    z_top = _blade_top(p, rr)
    z_bot = np.full(2 * n, float(p.backplate_thickness) - _ROOT_OVERLAP)
    V = np.concatenate([np.column_stack([poly, z_bot]), np.column_stack([poly, z_top])], 0)
    faces = [_ring_quads(2 * n, 0, 2 * n)]                      # side walls around the polygon
    # caps: polygon index i (left side, 0..n-1) pairs with 2n-1-i (right side)
    i = np.arange(n - 1)
    L0, L1, R0, R1 = i, i + 1, 2 * n - 1 - i, 2 * n - 2 - i
    for off in (0, 2 * n):
        faces.append(np.concatenate([np.stack([L0, L1, R1], 1), np.stack([L0, R1, R0], 1)], 0) + off)
    return V, np.concatenate(faces, 0)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
def build_mesh(params: ImpellerParams, *, handedness: str = "ccw", phase_deg: float = 0.0,
               n_camber: int = 24, n_seg: int = 96, with_hub: bool = True,
               with_shroud: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangle mesh of *params* -> ``(V, F, ids)`` with per-triangle part ids (PART_IDS).

    < 50 ms for the factory impeller (about 4-5k triangles).
    """
    if handedness not in ("ccw", "cw"):
        raise ValueError(f"handedness must be 'ccw' or 'cw' (got {handedness!r})")
    p = params
    tb = float(p.backplate_thickness)
    r_out = float(p.r2) + BACKPLATE_RIM_MM
    r_bore = float(p.hub_bore_diameter) / 2.0 if p.hub_bore_diameter else 0.0
    parts: list[tuple[np.ndarray, np.ndarray, int]] = []

    # backplate: top, bottom, outer side (+ bore wall)
    prof = [(r_out, 0.0), (r_out, tb)]
    if r_bore > 1e-9:
        prof = [(r_bore, 0.0)] + prof + [(r_bore, tb)]
        parts.append((*_revolve(prof, n_seg, closed=True, max_len=r_out / 4.0), PART_IDS["backplate"]))
    else:
        for z in (0.0, tb):
            parts.append((*_fan(r_out, z, n_seg), PART_IDS["backplate"]))
        parts.append((*_revolve(prof, n_seg, closed=False), PART_IDS["backplate"]))

    # hub boss
    if with_hub and p.hub_diameter > 0:
        z_hub = _hub_top(p)
        if z_hub > tb + 1e-6:
            prof = [(float(p.r_hub), tb), (float(p.r_hub), z_hub)]
            prof = prof + ([(r_bore, z_hub)] if r_bore > 1e-9 else [(0.0, z_hub)])
            parts.append((*_revolve(prof, n_seg, closed=False, max_len=p.r2 / 4.0), PART_IDS["hub"]))

    # blades
    sign = 1.0 if handedness == "ccw" else -1.0
    for i in range(int(p.blade_count)):
        theta0 = math.radians(phase_deg) + 2.0 * math.pi * i / p.blade_count
        parts.append((*_blade(p, theta0, sign, n_camber), PART_IDS["blade"]))

    # shroud
    if with_shroud and p.shroud:
        if _is_shrouded_v2(p):
            parts.append((*_revolve(_shroud_profile(p), n_seg, closed=True, max_len=p.r2 / 4.0), PART_IDS["shroud"]))
        else:
            ts = float(p.shroud_thickness)
            z0 = tb + float(p.blade_width)
            prof = [(float(p.r1), z0), (float(p.r2), z0), (float(p.r2), z0 + ts), (float(p.r1), z0 + ts)]
            parts.append((*_revolve(prof, n_seg, closed=True), PART_IDS["shroud"]))

    Vs, Fs, ids = [], [], []
    off = 0
    for V, F, pid in parts:
        Vs.append(V)
        Fs.append(F + off)
        ids.append(np.full(len(F), pid, np.int32))
        off += len(V)
    return (np.concatenate(Vs, 0).astype(np.float64), np.concatenate(Fs, 0).astype(np.int64),
            np.concatenate(ids, 0))


def _main(argv: list[str] | None = None) -> int:    # pragma: no cover - manual check
    import argparse
    import time

    from .params import SHROUDED_PARAMS_FACTORY

    ap = argparse.ArgumentParser(description="Build the fast mesh and report its size / timing")
    ap.add_argument("--params", default=None)
    ap.add_argument("--stl", default=None, help="write an STL (needs cad.save_stl)")
    ap.add_argument("--handedness", default="ccw")
    a = ap.parse_args(argv)
    p = ImpellerParams.load(a.params) if a.params else SHROUDED_PARAMS_FACTORY()
    t0 = time.perf_counter()
    V, F, ids = build_mesh(p, handedness=a.handedness)
    dt = time.perf_counter() - t0
    print(f"{len(V)} vertices, {len(F)} triangles in {1e3 * dt:.1f} ms; "
          f"parts: { {k: int(np.sum(ids == v)) for k, v in PART_IDS.items()} }")
    if a.stl:
        from .cad import save_stl
        save_stl(V, F, a.stl)
    return 0


if __name__ == "__main__":    # pragma: no cover
    raise SystemExit(_main())
