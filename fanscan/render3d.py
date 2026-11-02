"""Dependency-free software renderer for photo-like test images of impellers.

Pipeline: ``ImpellerParams`` -> CAD solid (cad.build_impeller) -> triangle mesh ->
pinhole camera -> z-buffer rasterisation (numpy) -> flat Lambert shading with a
small specular + "head-light" fill term -> optional blur / noise / clutter
background.

World frame = impeller frame (mm): z up, origin at the backplate bottom centre.
Image frame = OpenCV (x right, y down).  For the axial presets the camera's
``up`` is world +y, so world +y points *up* on screen and the v0.1 convention
(blade sweeps CCW in a y-up frame => clockwise on screen) is preserved.

Public API (see SPEC2.md, Module B):
    Camera, camera_preset, rasterize, shade, render, clutter_background,
    classify_faces, mesh_from_params, render_params, save

``render`` returns ``img`` (uint8 gray), or a tuple ``(img, depth)``,
``(img, ids)`` or ``(img, depth, ids)`` when ``return_depth`` and/or
``face_ids`` are given.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .params import ImpellerParams, SHROUDED_PARAMS_FACTORY

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PRESETS = ("axial_far", "axial_close", "tilt", "oblique", "side")
PART_IDS = {"backplate": 0, "hub": 1, "shroud": 2, "blade": 3, "marker": 4}
BACKPLATE_RIM_MM = 2.0          # must match cad.BACKPLATE_RIM_MM
_NEAR_MM = 1.0                  # triangles with a vertex closer than this are dropped
_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    position: np.ndarray            # (3,) mm, world frame
    target: np.ndarray              # (3,) look-at point
    up: np.ndarray                  # (3,) approximate up direction
    focal_px: float                 # pinhole focal length in px
    size: tuple[int, int] = (1200, 1200)   # (width, height)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.target = np.asarray(self.target, dtype=float).reshape(3)
        self.up = np.asarray(self.up, dtype=float).reshape(3)
        self.focal_px = float(self.focal_px)
        self.size = (int(self.size[0]), int(self.size[1]))

    # principal point at the centre of the pixel grid (pixel centres at integers)
    @property
    def cx(self) -> float:
        return (self.size[0] - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.size[1] - 1) / 2.0

    def rotation(self) -> np.ndarray:
        """3x3 world->camera rotation; rows = camera right, down, forward axes."""
        f = _unit(self.target - self.position)
        right = np.cross(f, self.up)
        if np.linalg.norm(right) < 1e-6:          # up parallel to the view: pick another
            for alt in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
                right = np.cross(f, np.asarray(alt))
                if np.linalg.norm(right) > 1e-6:
                    break
        right = _unit(right)
        down = np.cross(f, right)
        return np.stack([right, down, f])

    def intrinsics(self) -> np.ndarray:
        f = self.focal_px
        return np.array([[f, 0.0, self.cx], [0.0, f, self.cy], [0.0, 0.0, 1.0]])

    def extrinsics(self) -> np.ndarray:
        R = self.rotation()
        t = -R @ self.position
        return np.hstack([R, t[:, None]])

    def to_camera(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        return (P - self.position) @ self.rotation().T

    def project(self, P: np.ndarray) -> np.ndarray:
        """(n,3) world -> (n,3): pixel x, pixel y, depth (camera z, mm)."""
        Xc = self.to_camera(P)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.focal_px * Xc[:, 0] / z + self.cx
            y = self.focal_px * Xc[:, 1] / z + self.cy
        return np.stack([x, y, z], axis=1)

    def view_dir(self) -> np.ndarray:
        return _unit(self.target - self.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "K": self.intrinsics().tolist(),
            "R": self.rotation().tolist(),
            "t": (-self.rotation() @ self.position).tolist(),
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "focal_px": self.focal_px,
            "size": list(self.size),
        }


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------
def mesh_from_params(params: ImpellerParams, tolerance: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(V (n,3) float64, F (m,3) int64) for *params*; cached per (params, tolerance).

    Uses ``cad.mesh_from_params`` when the cad module provides it, otherwise
    tessellates ``cad.build_impeller(params)`` itself.
    """
    key = json.dumps({k: v for k, v in params.to_dict().items() if k != "meta"},
                     sort_keys=True, default=str) + f"|{tolerance}"
    hit = _MESH_CACHE.get(key)
    if hit is not None:
        return hit
    from . import cad  # local import: cadquery is slow to import

    fn = getattr(cad, "mesh_from_params", None)
    if fn is not None:
        V, F = fn(params, tolerance)
        V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
        F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    else:
        solid = cad.build_impeller(params).val()
        verts, tris = solid.tessellate(tolerance)
        V = np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float64).reshape(-1, 3)
        F = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    _MESH_CACHE[key] = (V, F)
    return V, F


def _z_levels(params: ImpellerParams) -> dict[str, float]:
    """Characteristic heights (mm) of the model built by cad.py.

    Uses ``cad.key_heights`` when the cad module provides it (same formulas),
    otherwise the local fallback below.
    """
    tb = float(params.backplate_thickness)
    b1, b2 = float(params.b1), float(params.b2)
    t = float(params.shroud_thickness)
    try:
        from . import cad
        kh = cad.key_heights(params)
    except (ImportError, AttributeError):
        kh = None
    if kh is not None:
        z_rim_top = float(kh.get("rim_top", tb))
        z_eye = float(kh.get("eye_ring_top", tb))
        hub_h = float(kh["hub_top"])
        z_top = float(kh["total_height"])
        return {"tb": tb, "z_rim_top": z_rim_top, "z_eye": z_eye, "hub_h": hub_h, "z_top": z_top}
    hub_h = float(params.hub_height) if params.hub_height is not None else b2 + tb
    if params.shroud and params.eye_diameter:
        cone = float(params.shroud_cone_height)
        inlet = float(params.shroud_inlet_height)
        z_rim_top = tb + b2 + t
        z_eye = tb + b2 + cone + max(inlet, t)
        hub_h = float(params.hub_height) if params.hub_height is not None else tb + b1
    elif params.shroud:
        z_rim_top = tb + b2 + t
        z_eye = z_rim_top
    else:
        z_rim_top = tb
        z_eye = tb
    return {"tb": tb, "z_rim_top": z_rim_top, "z_eye": z_eye, "hub_h": hub_h,
            "z_top": max(z_eye, z_rim_top, hub_h, tb + b1)}


def _shroud_underside(params: ImpellerParams, r: np.ndarray) -> np.ndarray:
    """Height above the backplate top of the shroud underside at radius r."""
    b2 = float(params.b2)
    if not params.shroud:
        return np.full_like(np.asarray(r, float), np.inf)
    if not params.eye_diameter:
        return np.full_like(np.asarray(r, float), b2)
    cone = float(params.shroud_cone_height)
    inlet = float(params.shroud_inlet_height)
    r2, re = params.r2, params.r_eye
    r = np.asarray(r, float)
    frac = np.clip((r2 - r) / max(r2 - re, 1e-9), 0.0, 1.0)
    z = b2 + cone * frac
    return np.where(r < re, b2 + cone + inlet, z)


def classify_faces(V: np.ndarray, F: np.ndarray, params: ImpellerParams) -> np.ndarray:
    """Per-triangle part id (see PART_IDS) from centroid position and normal.

    Blade side faces are the only non-axisymmetric surfaces: their normal has a
    tangential component, whereas every face of the backplate / hub / shroud is
    a surface of revolution (normal in the r-z plane).
    """
    tb = float(params.backplate_thickness)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    c = (A + B + C) / 3.0
    n = np.cross(B - A, C - A)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-12)[:, None]
    r = np.hypot(c[:, 0], c[:, 1])
    rhat = np.stack([c[:, 0], c[:, 1]], axis=1) / np.maximum(r, 1e-9)[:, None]
    n_xy = n[:, :2]
    n_xy_len = np.maximum(np.linalg.norm(n_xy, axis=1), 1e-9)
    tangential = np.abs(n_xy[:, 0] * rhat[:, 1] - n_xy[:, 1] * rhat[:, 0]) / n_xy_len
    b1, b2 = float(params.b1), float(params.b2)
    h_blade = b2 + (b1 - b2) * np.clip((params.r2 - r) / (params.r2 - params.r1), 0, 1)
    top = tb + np.minimum(h_blade, _shroud_underside(params, r))

    ids = np.full(len(F), PART_IDS["shroud"], dtype=np.int32)
    is_blade = ((np.abs(n[:, 2]) < 0.95) & (tangential > 0.12)
                & (c[:, 2] > tb + 0.2) & (c[:, 2] < top - 0.2)
                & (r > params.r_hub + 0.2))
    ids[is_blade] = PART_IDS["blade"]
    ids[(c[:, 2] <= tb + 0.2)] = PART_IDS["backplate"]
    ids[(r <= params.r_hub + 0.2) & (c[:, 2] > tb + 0.2)] = PART_IDS["hub"]
    return ids


def marker_box(params: ImpellerParams, azimuth_deg: float = 270.0,
               size_mm: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small bright cube sitting on the shroud rim at *azimuth_deg*.

    Returns (V, F, centre_xyz).  The box straddles the rim top edge so it is
    visible both axially (square on the rim) and from the side (bump).
    """
    s = float(size_mm) if size_mm else max(4.0, 0.025 * params.outer_diameter)
    z0 = _z_levels(params)["z_rim_top"] - 0.5
    r_in, r_out = params.r2 - s, params.r2
    a = math.radians(azimuth_deg)
    ca, sa = math.cos(a), math.sin(a)
    # box axes: radial (ca, sa), tangential (-sa, ca), z
    corners = []
    for rr in (r_in, r_out):
        for tt in (-s / 2, s / 2):
            for zz in (z0, z0 + s):
                corners.append((rr * ca - tt * sa, rr * sa + tt * ca, zz))
    V = np.array(corners, dtype=float)
    # index = (ri * 2 + ti) * 2 + zi
    def idx(ri, ti, zi):
        return (ri * 2 + ti) * 2 + zi
    quads = [
        (idx(0, 0, 0), idx(0, 1, 0), idx(0, 1, 1), idx(0, 0, 1)),   # inner
        (idx(1, 0, 0), idx(1, 0, 1), idx(1, 1, 1), idx(1, 1, 0)),   # outer
        (idx(0, 0, 0), idx(0, 0, 1), idx(1, 0, 1), idx(1, 0, 0)),   # -t
        (idx(0, 1, 0), idx(1, 1, 0), idx(1, 1, 1), idx(0, 1, 1)),   # +t
        (idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0)),   # bottom
        (idx(0, 0, 1), idx(0, 1, 1), idx(1, 1, 1), idx(1, 0, 1)),   # top
    ]
    F = []
    for q in quads:
        F.append((q[0], q[1], q[2]))
        F.append((q[0], q[2], q[3]))
    centre = np.array([(r_in + r_out) / 2 * ca, (r_in + r_out) / 2 * sa, z0 + s / 2])
    return V, np.asarray(F, dtype=np.int64), centre


# ---------------------------------------------------------------------------
# rasteriser
# ---------------------------------------------------------------------------
@dataclass
class Raster:
    depth: np.ndarray     # (H,W) float32, +inf where nothing was hit
    tri: np.ndarray       # (H,W) int32 triangle index, -1 = background
    bary: np.ndarray      # (H,W,3) float32 barycentric coords of the hit
    proj: np.ndarray      # (n,3) projected vertices (x, y, depth)

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0

    def world_points(self, V: np.ndarray, F: np.ndarray) -> np.ndarray:
        """(H,W,3) world coordinates of the surface hit at each pixel (nan = bg)."""
        out = np.full(self.tri.shape + (3,), np.nan, dtype=np.float64)
        m = self.mask
        t = self.tri[m]
        P = V[F[t]]                                 # (k,3,3)
        out[m] = np.einsum("kj,kjd->kd", self.bary[m].astype(np.float64), P)
        return out


def rasterize(V: np.ndarray, F: np.ndarray, cam: Camera, *,
              pixel_budget: int = 2_500_000) -> Raster:
    """z-buffer rasterisation of a triangle mesh (two-sided, flat depth test).

    Triangles are grouped by bounding-box size (log2 buckets) and processed in
    vectorised chunks whose candidate-pixel count stays under *pixel_budget*.
    """
    W, H = cam.size
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    P = cam.project(V)
    px, py, pd = P[:, 0], P[:, 1], P[:, 2]

    tx, ty, td = px[F], py[F], pd[F]                           # (m,3)
    xa, ya, xb, yb, xc, yc = tx[:, 0], ty[:, 0], tx[:, 1], ty[:, 1], tx[:, 2], ty[:, 2]
    area = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
    ok = (td.min(axis=1) > _NEAR_MM) & (np.abs(area) > 1e-9)
    fxmin, fxmax = np.floor(tx.min(axis=1)), np.ceil(tx.max(axis=1))
    fymin, fymax = np.floor(ty.min(axis=1)), np.ceil(ty.max(axis=1))
    ok &= (fxmax >= 0) & (fxmin <= W - 1) & (fymax >= 0) & (fymin <= H - 1)
    ok &= np.isfinite(fxmin) & np.isfinite(fxmax) & np.isfinite(fymin) & np.isfinite(fymax)
    with np.errstate(invalid="ignore"):
        xmin = np.clip(np.nan_to_num(fxmin, nan=0), 0, W - 1).astype(np.int64)
        xmax = np.clip(np.nan_to_num(fxmax, nan=0), 0, W - 1).astype(np.int64)
        ymin = np.clip(np.nan_to_num(fymin, nan=0), 0, H - 1).astype(np.int64)
        ymax = np.clip(np.nan_to_num(fymax, nan=0), 0, H - 1).astype(np.int64)
    bw, bh = xmax - xmin + 1, ymax - ymin + 1

    depth = np.full(W * H, np.inf, dtype=np.float32)
    tri = np.full(W * H, -1, dtype=np.int32)
    bary = np.zeros((W * H, 3), dtype=np.float32)

    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return Raster(depth.reshape(H, W), tri.reshape(H, W), bary.reshape(H, W, 3), P)
    kx = np.ceil(np.log2(bw[cand])).astype(np.int64)
    ky = np.ceil(np.log2(bh[cand])).astype(np.int64)
    key = kx * 64 + ky
    order = np.argsort(key, kind="stable")
    cand, key = cand[order], key[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    groups = np.split(cand, bounds)

    for g in groups:
        BW, BH = int(bw[g].max()), int(bh[g].max())
        per = max(1, pixel_budget // (BW * BH))
        for s in range(0, len(g), per):
            sel = g[s:s + per]
            n = len(sel)
            gx = xmin[sel][:, None, None] + np.arange(BW)[None, None, :]      # (n,1,BW)
            gy = ymin[sel][:, None, None] + np.arange(BH)[None, :, None]      # (n,BH,1)
            Xa, Ya = xa[sel][:, None, None], ya[sel][:, None, None]
            Xb, Yb = xb[sel][:, None, None], yb[sel][:, None, None]
            Xc, Yc = xc[sel][:, None, None], yc[sel][:, None, None]
            w_a = (Xc - Xb) * (gy - Yb) - (Yc - Yb) * (gx - Xb)
            w_b = (Xa - Xc) * (gy - Yc) - (Ya - Yc) * (gx - Xc)
            w_c = (Xb - Xa) * (gy - Ya) - (Yb - Ya) * (gx - Xa)
            inside = (((w_a >= 0) & (w_b >= 0) & (w_c >= 0))
                      | ((w_a <= 0) & (w_b <= 0) & (w_c <= 0)))
            inside &= (gx <= xmax[sel][:, None, None]) & (gy <= ymax[sel][:, None, None])
            ti, iy, ix = np.nonzero(inside)
            if len(ti) == 0:
                continue
            ar = area[sel][ti]
            la = w_a[ti, iy, ix] / ar
            lb = w_b[ti, iy, ix] / ar
            lc = w_c[ti, iy, ix] / ar
            dsel = td[sel]
            d = (la * dsel[ti, 0] + lb * dsel[ti, 1] + lc * dsel[ti, 2]).astype(np.float32)
            lin = (ymin[sel][ti] + iy) * W + (xmin[sel][ti] + ix)
            np.minimum.at(depth, lin, d)
            win = d <= depth[lin]
            lw = lin[win]
            tri[lw] = sel[ti[win]].astype(np.int32)
            bary[lw, 0] = la[win]
            bary[lw, 1] = lb[win]
            bary[lw, 2] = lc[win]

    return Raster(depth.reshape(H, W), tri.reshape(H, W), bary.reshape(H, W, 3), P)


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------
def face_shading(V: np.ndarray, F: np.ndarray, cam: Camera, *,
                 light_dir=(0.3, -0.4, 1.0), ambient: float = 0.35,
                 albedo: float | np.ndarray = 0.85, fill_light: float = 0.25,
                 specular: float = 0.12, shininess: float = 24.0) -> np.ndarray:
    """Per-triangle gray level in [0, ~1.1] (flat, two-sided Lambert + fill + specular)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    c = (A + B + C) / 3.0
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    v = cam.position[None, :] - c
    v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
    flip = np.einsum("ij,ij->i", n, v) < 0
    n[flip] *= -1.0                                   # two-sided: face the camera
    L = _unit(light_dir)
    ndl = np.clip(n @ L, 0.0, 1.0)
    ndv = np.clip(np.einsum("ij,ij->i", n, v), 0.0, 1.0)
    h = L[None, :] + v
    h /= np.maximum(np.linalg.norm(h, axis=1), 1e-12)[:, None]
    ndh = np.clip(np.einsum("ij,ij->i", n, h), 0.0, 1.0)
    diffuse = (1.0 - fill_light) * ndl + fill_light * ndv
    alb = np.broadcast_to(np.asarray(albedo, dtype=float), (len(F),))
    return alb * (ambient + (1.0 - ambient) * diffuse) + specular * ndh ** shininess


def pixel_shading(V: np.ndarray, F: np.ndarray, cam: Camera, raster: Raster, *,
                  light_dir=(0.3, -0.4, 1.0), ambient: float = 0.35,
                  albedo: float | np.ndarray = 0.85, fill_light: float = 0.25,
                  specular: float = 0.12, shininess: float = 24.0) -> np.ndarray:
    """Per-pixel gray level for the covered pixels (flat normals, per-pixel view
    vector -> no sliver artefacts on flat faces, smooth specular highlights).

    Returns a 1-D array aligned with ``raster.mask.nonzero()``.
    """
    m = raster.mask
    t = raster.tri[m]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = np.cross(B - A, C - A)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    n = N[t]
    P = np.einsum("kj,kjd->kd", raster.bary[m].astype(np.float64), V[F[t]])
    v = cam.position[None, :] - P
    v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
    ndv = np.einsum("ij,ij->i", n, v)
    n = np.where((ndv < 0)[:, None], -n, n)            # two-sided: face the camera
    ndv = np.abs(ndv)
    L = _unit(light_dir)
    ndl = np.clip(n @ L, 0.0, 1.0)
    h = L[None, :] + v
    h /= np.maximum(np.linalg.norm(h, axis=1), 1e-12)[:, None]
    ndh = np.clip(np.einsum("ij,ij->i", n, h), 0.0, 1.0)
    diffuse = (1.0 - fill_light) * ndl + fill_light * np.clip(ndv, 0.0, 1.0)
    alb = np.broadcast_to(np.asarray(albedo, dtype=float), (len(F),))[t]
    return alb * (ambient + (1.0 - ambient) * diffuse) + specular * ndh ** shininess


def shade(V: np.ndarray, F: np.ndarray, cam: Camera, raster: Raster, *,
          background: np.ndarray | float = 0.12, **light_kw) -> np.ndarray:
    """Float image (H,W) in [0,1]: shaded surface over *background*."""
    W, H = cam.size
    if isinstance(background, np.ndarray):
        bg = background
        if bg.ndim == 3:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        is_u8 = bg.dtype == np.uint8
        bg = bg.astype(np.float32)
        if bg.shape != (H, W):
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        if is_u8:
            bg = bg / 255.0
    else:
        bg = np.full((H, W), float(background), dtype=np.float32)
    img = bg.copy()
    m = raster.mask
    if m.any():
        img[m] = pixel_shading(V, F, cam, raster, **light_kw).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _finish(img01: np.ndarray, noise_sigma: float, blur_sigma: float, seed: int) -> np.ndarray:
    img = img01.astype(np.float32) * 255.0
    if blur_sigma and blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(blur_sigma))
    if noise_sigma and noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, float(noise_sigma), img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def render(V: np.ndarray, F: np.ndarray, cam: Camera, *,
           light_dir=(0.3, -0.4, 1.0), ambient: float = 0.35,
           albedo: float | np.ndarray = 0.85,
           background: np.ndarray | float = 0.12,
           noise_sigma: float = 0.0, blur_sigma: float = 0.0, seed: int = 0,
           return_depth: bool = False, face_ids: np.ndarray | None = None,
           fill_light: float = 0.25, specular: float = 0.12, shininess: float = 24.0,
           pixel_budget: int = 2_500_000):
    """Render mesh (V, F) through *cam* to a uint8 gray image.

    Returns ``img``; ``(img, depth)`` if ``return_depth``; ``(img, ids)`` if
    ``face_ids`` (per-triangle int array) is given (ids image is int32, -1 for
    background); ``(img, depth, ids)`` if both.
    """
    raster = rasterize(V, F, cam, pixel_budget=pixel_budget)
    img01 = shade(V, F, cam, raster, background=background, light_dir=light_dir,
                  ambient=ambient, albedo=albedo, fill_light=fill_light,
                  specular=specular, shininess=shininess)
    img = _finish(img01, noise_sigma, blur_sigma, seed)
    out: list[Any] = [img]
    if return_depth:
        out.append(raster.depth.copy())
    if face_ids is not None:
        face_ids = np.asarray(face_ids)
        ids = np.full(raster.tri.shape, -1, dtype=np.int32)
        m = raster.mask
        ids[m] = face_ids[raster.tri[m]]
        out.append(ids)
    return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------------------
# procedural clutter background
# ---------------------------------------------------------------------------
def clutter_background(size: tuple[int, int] = (1200, 1200), seed: int = 0) -> np.ndarray:
    """Desk-like clutter: light wood tone with grain, dark rectangles (keyboard /
    mouse / monitor stand), thin dark cable curves and a bright blob.  uint8 gray.
    """
    W, H = int(size[0]), int(size[1])
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # wood: base tone + gentle lighting gradient + low-frequency blotches + grain streaks
    ang = rng.uniform(0, 2 * np.pi)
    grad = (xx * math.cos(ang) + yy * math.sin(ang)) / max(W, H)
    img = 168.0 + 22.0 * (grad - 0.5)
    small = rng.normal(0, 1, (8, 8)).astype(np.float32)
    img += 9.0 * cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    grain_ang = rng.uniform(-0.25, 0.25)
    u = (xx * math.cos(grain_ang) + yy * math.sin(grain_ang))
    grain = np.zeros_like(img)
    for k in range(3):
        f = rng.uniform(0.08, 0.35)
        grain += rng.uniform(2, 5) * np.sin(u * f + rng.uniform(0, 6.28) + 0.6 * np.sin(u * f * 0.13))
    img += grain
    img = img.astype(np.float32)

    def rot_rect(cx, cy, w, h, theta):
        c, s = math.cos(theta), math.sin(theta)
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        pts = pts @ np.array([[c, -s], [s, c]]).T + [cx, cy]
        return pts.astype(np.int32)

    # dark rectangles (keyboard-ish large one near an edge, a few small ones)
    n_rect = int(rng.integers(3, 6))
    for i in range(n_rect):
        big = i == 0
        w = rng.uniform(0.35, 0.6) * W if big else rng.uniform(0.06, 0.2) * W
        h = rng.uniform(0.12, 0.25) * H if big else rng.uniform(0.05, 0.15) * H
        side = rng.integers(0, 4)
        cx = rng.uniform(0.1, 0.9) * W
        cy = rng.uniform(0.1, 0.9) * H
        if side == 0:
            cy = rng.uniform(-0.05, 0.15) * H
        elif side == 1:
            cy = rng.uniform(0.85, 1.05) * H
        elif side == 2:
            cx = rng.uniform(-0.05, 0.15) * W
        else:
            cx = rng.uniform(0.85, 1.05) * W
        tone = float(rng.uniform(35, 75))
        theta = rng.uniform(-0.4, 0.4)
        cv2.fillPoly(img, [rot_rect(cx, cy, w, h, theta)], tone, lineType=cv2.LINE_AA)
        if big:   # key rows
            nk = int(rng.integers(4, 7))
            for k in range(nk):
                for j in range(int(rng.integers(8, 14))):
                    kx = cx - w / 2 + (j + 0.5) * w / 13
                    ky = cy - h / 2 + (k + 0.5) * h / nk
                    cv2.fillPoly(img, [rot_rect(kx, ky, w / 15, h / (nk + 1.5), theta)],
                                 tone + 25, lineType=cv2.LINE_AA)
    # a sheet of paper
    cv2.fillPoly(img, [rot_rect(rng.uniform(0, W), rng.uniform(0, H), 0.25 * W, 0.18 * H,
                                rng.uniform(-0.5, 0.5))], float(rng.uniform(205, 225)),
                 lineType=cv2.LINE_AA)
    # cables: smooth random curves
    for _ in range(int(rng.integers(2, 5))):
        k = int(rng.integers(4, 7))
        ctrl = np.stack([rng.uniform(-0.1, 1.1, k) * W, rng.uniform(-0.1, 1.1, k) * H], 1)
        t = np.linspace(0, 1, k)
        tt = np.linspace(0, 1, 200)
        # Catmull-Rom-ish via cubic polynomial fit on each coordinate
        deg = min(3, k - 1)
        cx = np.polyval(np.polyfit(t, ctrl[:, 0], deg), tt)
        cy = np.polyval(np.polyfit(t, ctrl[:, 1], deg), tt)
        pts = np.stack([cx, cy], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, float(rng.uniform(20, 45)),
                      int(rng.integers(3, 7)), lineType=cv2.LINE_AA)
    # bright blob (lamp reflection / white object)
    cv2.ellipse(img, (int(rng.uniform(0, W)), int(rng.uniform(0, H))),
                (int(rng.uniform(0.03, 0.08) * W), int(rng.uniform(0.02, 0.06) * H)),
                float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(238, 252)), -1,
                lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# camera presets
# ---------------------------------------------------------------------------
def camera_preset(name: str, params: ImpellerParams, size: tuple[int, int] = (1200, 1200), *,
                  distance: float | None = None, focal_px: float | None = None,
                  tilt_deg: float = 15.0, elevation_deg: float | None = None,
                  azimuth_deg: float | None = None, fill: float | None = None,
                  target: Any = None, up: Any = None, **_ignored) -> Camera:
    """Camera for one of the named presets.

    'axial_far'   on axis, distance 6*D2 above the backplate, telephoto
    'axial_close' on axis, distance 1.2*D2, phone-like focal = 0.9*width
    'tilt'        axis tilted by tilt_deg (default 15), distance 3*D2
    'oblique'     elevation 40 deg, distance 2.5*D2, aimed at the mid height
    'side'        elevation 5 deg, distance 2.5*D2, aimed at the near rim gap
    Keyword overrides: distance, focal_px, tilt_deg, elevation_deg, azimuth_deg
    (direction the camera is displaced towards, deg from +x; default 270 = -y),
    fill (rim radius as a fraction of min(size)), target, up.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {PRESETS}")
    W, H = int(size[0]), int(size[1])
    D2, r2 = float(params.outer_diameter), float(params.r2)
    zl = _z_levels(params)
    tb, z_top = zl["tb"], zl["z_top"]
    az = math.radians(270.0 if azimuth_deg is None else azimuth_deg)
    mn = min(W, H)

    if name == "axial_far":
        d = 6.0 * D2 if distance is None else float(distance)
        tgt = np.array([0.0, 0.0, tb]) if target is None else np.asarray(target, float)
        pos = tgt + np.array([0.0, 0.0, d])
        upv = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, float)
        f = (fill or 0.42) * mn * d / (r2 + BACKPLATE_RIM_MM) if focal_px is None else float(focal_px)
    elif name == "axial_close":
        d = 1.2 * D2 if distance is None else float(distance)
        tgt = np.array([0.0, 0.0, tb]) if target is None else np.asarray(target, float)
        pos = tgt + np.array([0.0, 0.0, d])
        upv = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, float)
        f = (fill * mn * d / r2 if fill else 0.9 * W) if focal_px is None else float(focal_px)
    elif name == "tilt":
        d = 3.0 * D2 if distance is None else float(distance)
        tgt = np.array([0.0, 0.0, tb]) if target is None else np.asarray(target, float)
        t = math.radians(tilt_deg)
        pos = tgt + d * np.array([math.cos(az) * math.sin(t), math.sin(az) * math.sin(t), math.cos(t)])
        upv = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, float)
        f = (fill or 0.40) * mn * d / r2 if focal_px is None else float(focal_px)
    elif name == "oblique":
        d = 2.5 * D2 if distance is None else float(distance)
        el = math.radians(40.0 if elevation_deg is None else elevation_deg)
        tgt = np.array([0.0, 0.0, 0.5 * (tb + z_top)]) if target is None else np.asarray(target, float)
        pos = tgt + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        upv = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
        f = (fill or 0.40) * mn * d / r2 if focal_px is None else float(focal_px)
    else:  # side
        d = 2.5 * D2 if distance is None else float(distance)
        el = math.radians(5.0 if elevation_deg is None else elevation_deg)
        gap_z = tb + 0.5 * float(params.b2)
        tgt = (np.array([r2 * math.cos(az), r2 * math.sin(az), gap_z]) if target is None
               else np.asarray(target, float))
        pos = tgt + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        upv = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
        f = (fill or 0.42) * mn * d / r2 if focal_px is None else float(focal_px)
    return Camera(position=pos, target=tgt, up=upv, focal_px=f, size=(W, H))


# ---------------------------------------------------------------------------
# truth helpers
# ---------------------------------------------------------------------------
def _ellipse_dict(cam: Camera, r: float, z: float, n: int = 360) -> dict[str, Any] | None:
    """cv2.fitEllipse of the projected circle (radius r at height z, axis = z)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], 1)
    p = cam.project(P)
    if not np.all(p[:, 2] > _NEAR_MM) or not np.all(np.isfinite(p[:, :2])):
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(p[:, :2].astype(np.float32))
    if w >= h:
        a, b, major = w / 2, h / 2, ang
    else:
        a, b, major = h / 2, w / 2, ang + 90.0
    return {"center": [float(cx), float(cy)], "axes": [float(a), float(b)],
            "angle_deg": float(major % 180.0), "r_mm": float(r), "z_mm": float(z),
            "cv2": [[float(cx), float(cy)], [float(w), float(h)], float(ang)]}


def ellipse_mask(e: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    """Boolean (H,W) mask of pixels inside an ellipse dict from ``_ellipse_dict``."""
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = e["center"]
    a, b = e["axes"]
    th = math.radians(e["angle_deg"])
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def _px_per_mm(cam: Camera, z: float) -> float:
    d = cam.project(np.array([[0.0, 0.0, z]]))[0, 2]
    return float(cam.focal_px / d) if d > 0 else float("nan")


# ---------------------------------------------------------------------------
# render_params
# ---------------------------------------------------------------------------
def render_params(params: ImpellerParams, preset: str = "axial_close", *,
                  size: tuple[int, int] = (1200, 1200), clutter: bool = False,
                  background: np.ndarray | float | None = None,
                  noise_sigma: float = 0.0, blur_sigma: float = 0.8, seed: int = 0,
                  marker: bool = False, marker_deg: float = 270.0,
                  marker_size_mm: float | None = None, tolerance: float = 0.5,
                  mesh: tuple[np.ndarray, np.ndarray] | None = None,
                  light_dir=(0.3, -0.4, 1.0), ambient: float = 0.35, albedo: float = 0.85,
                  fill_light: float = 0.25, specular: float = 0.12, shininess: float = 24.0,
                  **cam_kw) -> tuple[np.ndarray, dict[str, Any]]:
    """Render *params* with a named camera preset -> (uint8 gray image, truth dict).

    truth: camera (K/R/t/...), px_per_mm at the backplate plane on axis,
    center_xy (projected axis point at the backplate plane), projected rim /
    eye / backplate / hub / bore ellipses, visible blade fraction through the
    eye (from the part-id image), marker position, params.to_dict().
    Extra keyword arguments go to ``camera_preset`` (tilt_deg, azimuth_deg, ...).
    """
    t0 = time.perf_counter()
    V, F = mesh if mesh is not None else mesh_from_params(params, tolerance)
    ids = classify_faces(V, F, params)
    alb = np.full(len(F), float(albedo))
    marker_info = None
    if marker:
        Vm, Fm, cm = marker_box(params, marker_deg, marker_size_mm)
        F = np.vstack([F, Fm + len(V)])
        V = np.vstack([V, Vm])
        ids = np.concatenate([ids, np.full(len(Fm), PART_IDS["marker"], np.int32)])
        alb = np.concatenate([alb, np.full(len(Fm), 1.6)])    # saturated bright
        marker_info = {"azimuth_deg": float(marker_deg), "xyz": cm.tolist()}
    cam = camera_preset(preset, params, size, **cam_kw)
    W, H = cam.size
    if background is None:
        background = clutter_background((W, H), seed) if clutter else 0.12
    t1 = time.perf_counter()
    raster = rasterize(V, F, cam)
    t2 = time.perf_counter()
    img01 = shade(V, F, cam, raster, background=background, light_dir=light_dir,
                  ambient=ambient, albedo=alb, fill_light=fill_light,
                  specular=specular, shininess=shininess)
    img = _finish(img01, noise_sigma, blur_sigma, seed)
    id_img = np.full((H, W), -1, dtype=np.int32)
    id_img[raster.mask] = ids[raster.tri[raster.mask]]
    if marker_info is not None:
        marker_info["px"] = cam.project(np.asarray(marker_info["xyz"])[None, :])[0, :2].tolist()
        marker_info["visible_pixels"] = int(np.sum(id_img == PART_IDS["marker"]))

    # --- geometry truth ---------------------------------------------------
    zl = _z_levels(params)
    tb = zl["tb"]
    rim_shroud = _ellipse_dict(cam, params.r2, zl["z_rim_top"]) if params.shroud else None
    rim_under = (_ellipse_dict(cam, params.r2, zl["z_rim_top"] - float(params.shroud_thickness))
                 if params.shroud else None)
    rim_plate = _ellipse_dict(cam, params.r2 + BACKPLATE_RIM_MM, tb)
    if rim_shroud is not None and (rim_plate is None or rim_shroud["axes"][0] >= rim_plate["axes"][0]):
        rim, rim_source = rim_shroud, "shroud"
    else:
        rim, rim_source = rim_plate, "backplate"
    eye = _ellipse_dict(cam, params.r_eye, zl["z_eye"]) if params.shroud else \
        _ellipse_dict(cam, params.r1, tb)
    hub = _ellipse_dict(cam, params.r_hub, zl["hub_h"])
    bore = _ellipse_dict(cam, params.hub_bore_diameter / 2.0, zl["hub_h"]) \
        if params.hub_bore_diameter > 0 else None
    centre = cam.project(np.array([[0.0, 0.0, tb]]))[0]

    # --- visible blade fraction through the eye ----------------------------
    blade_px = id_img == PART_IDS["blade"]
    r_vis_root = r_vis_any = float("nan")
    n_in_eye = 0
    if eye is not None and blade_px.any():
        in_eye = ellipse_mask(eye, (W, H)) & blade_px
        n_in_eye = int(in_eye.sum())
        if n_in_eye:
            Pw = raster.world_points(V, F)
            rw = np.hypot(Pw[..., 0], Pw[..., 1])
            zw = Pw[..., 2]
            r_vis_any = float(np.nanmax(rw[in_eye]))
            root = in_eye & (zw < tb + max(0.5, 0.15 * float(params.b1)))
            r_vis_root = float(np.nanmax(rw[root])) if root.any() else float(params.r1)
    span = max(params.r2 - params.r1, 1e-9)
    vis_frac = float(np.clip((r_vis_root - params.r1) / span, 0.0, 1.0)) \
        if np.isfinite(r_vis_root) else 0.0
    t3 = time.perf_counter()

    truth: dict[str, Any] = {
        "preset": preset,
        "size": [W, H],
        "camera": cam.to_dict(),
        "px_per_mm": _px_per_mm(cam, tb),
        "px_per_mm_rim": _px_per_mm(cam, zl["z_rim_top"]),
        "px_per_mm_eye": _px_per_mm(cam, zl["z_eye"]),
        "center_xy": [float(centre[0]), float(centre[1])],
        "axis_px": {k: cam.project(np.array([[0.0, 0.0, z]]))[0, :2].tolist()
                    for k, z in (("z0", 0.0), ("backplate", tb), ("rim", zl["z_rim_top"]),
                                 ("eye", zl["z_eye"]), ("top", zl["z_top"]))},
        # rim_ellipse = outer silhouette circle in an axial view (shroud rim top edge, or the
        # backplate edge for open impellers).  NOTE: in tilted/oblique views the outer
        # silhouette is the union of the shroud rim and the backplate rim; compare detected
        # rim *edges* (not the whole silhouette) with rim_shroud_ellipse / rim_underside_ellipse.
        "rim_ellipse": rim, "rim_source": rim_source, "rim_shroud_ellipse": rim_shroud,
        "rim_underside_ellipse": rim_under, "backplate_ellipse": rim_plate,
        "eye_ellipse": eye, "hub_ellipse": hub, "bore_ellipse": bore,
        "z_levels": zl,
        "visible_blade_fraction": vis_frac,
        "r_vis_root_mm": r_vis_root, "r_vis_any_mm": r_vis_any,
        "blade_pixels_in_eye": n_in_eye,
        "marker": marker_info,
        "part_ids": dict(PART_IDS),
        "part_pixel_counts": {k: int(np.sum(id_img == v)) for k, v in PART_IDS.items()},
        "n_triangles": int(len(F)),
        "timing_s": {"mesh+ids": t1 - t0, "rasterize": t2 - t1, "shade+truth": t3 - t2,
                     "total": t3 - t0},
        "clutter": bool(clutter), "noise_sigma": float(noise_sigma),
        "blur_sigma": float(blur_sigma), "seed": int(seed),
        "params": params.to_dict(),
    }
    truth["_id_image"] = id_img          # not JSON-serialisable; stripped by truth_json
    truth["_depth"] = raster.depth
    return img, truth


def truth_json(truth: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable copy of a truth dict (drops the ``_``-prefixed arrays)."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    return conv(truth)


def save(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render all camera presets of an impeller")
    ap.add_argument("--out-dir", default="render3d_out")
    ap.add_argument("--params", default=None, help="params JSON (default: SHROUDED_PARAMS_FACTORY)")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--clutter", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--tilt-deg", type=float, default=15.0)
    ap.add_argument("--marker", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    params = ImpellerParams.load(a.params) if a.params else SHROUDED_PARAMS_FACTORY()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    V, F = mesh_from_params(params)
    print(f"mesh: {len(V)} vertices, {len(F)} triangles ({time.perf_counter() - t0:.2f} s)")
    for name in a.presets.split(","):
        name = name.strip()
        img, truth = render_params(params, name, size=(a.size, a.size), clutter=a.clutter,
                                   noise_sigma=a.noise, blur_sigma=a.blur, seed=a.seed,
                                   marker=a.marker, tilt_deg=a.tilt_deg)
        save(out / f"{name}.png", img)
        (out / f"{name}.truth.json").write_text(json.dumps(truth_json(truth), indent=1))
        tm = truth["timing_s"]
        print(f"{name:12s} raster {tm['rasterize']:.2f} s, total {tm['total']:.2f} s, "
              f"vis_frac {truth['visible_blade_fraction']:.2f}, "
              f"rim a/b {truth['rim_ellipse']['axes'][0]:.0f}/{truth['rim_ellipse']['axes'][1]:.0f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
