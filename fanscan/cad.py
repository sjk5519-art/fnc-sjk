"""CAD model of a centrifugal impeller (cadquery) from ``ImpellerParams``.

Geometry (z up, backplate on the bottom, all lengths in mm):

* backplate: disc r = r2 + 2 mm, z in [0, tb]
* hub:       cylinder r = r_hub, from z = 0, height = hub_height or (b + tb)
             (b = blade_width in open mode, b1 = blade_width_inlet in shrouded mode)
* blades:    Z extruded ``geometry.offset_camber`` polygons starting at z = tb,
             blade i rotated by 2*pi*i/Z (camber line sweeps toward +theta, CCW, as r grows)
* shroud (v0.1, ``shroud=True`` and ``eye_diameter is None``):
             flat ring r1..r2 of ``shroud_thickness`` sitting on top of the blades
* shroud (v0.2, ``shroud=True`` and ``eye_diameter`` set) -- "production-type" closed impeller:
             blade height tapers linearly h(r) = b2 + (b1 - b2) * (r2 - r) / (r2 - r1); a conical
             front shroud whose underside rises from z = tb + b2 at r2 to z = tb + b2 + cone at
             r_eye, an inlet ring (cylinder r in [r_eye, r_eye + ts]) on top of the cone, and a
             through-bore of ``hub_bore_diameter``.  The blade top is trimmed wherever the shroud
             underside is lower than the tapered height, so the blade always ends exactly on the
             shroud underside under the shroud and follows the taper inside the eye.

Everything is unioned into one solid.  v0.1 behaviour is unchanged for non-shrouded params.
"""
from __future__ import annotations

import math
import struct
import time
from pathlib import Path

import cadquery as cq
import numpy as np

from . import geometry
from .params import DEFAULT_PARAMS, ImpellerParams

BACKPLATE_RIM_MM = 2.0      # backplate radius = r2 + this
CAMBER_POINTS = 64          # >= 48 per SPEC so the blade polygon is smooth
SHROUD_OVERLAP_MM = 0.3     # blade top is pushed this far *into* the shroud so the union is robust
ROOT_OVERLAP_MM = 0.5       # (shrouded mode) blade root starts this far inside the backplate


def _is_shrouded_v2(params: ImpellerParams) -> bool:
    """True for the v0.2 cone + inlet-ring shroud (v0.1 flat ring otherwise)."""
    return bool(params.shroud) and params.eye_diameter is not None


def _hub_height(params: ImpellerParams) -> float:
    """Total hub height measured from z = 0 (hub_height, or blade height + tb)."""
    if params.hub_height is not None:
        return float(params.hub_height)
    b = params.b1 if _is_shrouded_v2(params) else params.blade_width
    return float(b + params.backplate_thickness)


# ---------------------------------------------------------------------------
# Analytic profile helpers (also useful for the renderer / analyzer modules)
# ---------------------------------------------------------------------------

def blade_height(params: ImpellerParams, r: np.ndarray | float) -> np.ndarray | float:
    """Untrimmed blade height h(r) above the backplate top face.

    Linear taper b1 -> b2 over [r1, r2] in shrouded mode (clamped outside), constant
    ``blade_width`` otherwise.
    """
    if not _is_shrouded_v2(params):
        return np.full_like(np.asarray(r, float), params.blade_width) if np.ndim(r) else params.blade_width
    r_arr = np.asarray(r, dtype=float)
    t = np.clip((params.r2 - r_arr) / (params.r2 - params.r1), 0.0, 1.0)
    h = params.b2 + (params.b1 - params.b2) * t
    return h if np.ndim(r) else float(h)


def shroud_underside_z(params: ImpellerParams, r: np.ndarray | float) -> np.ndarray | float:
    """Absolute z of the shroud underside at radius r (valid for r in [r_eye, r2])."""
    tb = params.backplate_thickness
    if not _is_shrouded_v2(params):
        z = tb + params.blade_width
        return np.full_like(np.asarray(r, float), z) if np.ndim(r) else z
    r_arr = np.asarray(r, dtype=float)
    frac = np.clip((params.r2 - r_arr) / max(params.r2 - params.r_eye, 1e-9), 0.0, 1.0)
    z = tb + params.b2 + params.shroud_cone_height * frac
    return z if np.ndim(r) else float(z)


def key_heights(params: ImpellerParams) -> dict[str, float]:
    """Absolute z of the main horizontal features (mm).  Handy for pose / scale work."""
    tb = float(params.backplate_thickness)
    ts = float(params.shroud_thickness)
    out = {"backplate_top": tb, "hub_top": _hub_height(params)}
    if _is_shrouded_v2(params):
        z_rim = tb + params.b2
        z_cone_top = z_rim + params.shroud_cone_height
        ring_top = z_cone_top + max(params.shroud_inlet_height, ts)
        out.update(rim_underside=z_rim, rim_top=z_rim + ts, eye_underside=z_cone_top,
                   eye_ring_top=ring_top, blade_top_le=tb + params.b1,
                   total_height=max(ring_top, out["hub_top"], tb + params.b1))
    elif params.shroud:
        z_rim = tb + params.blade_width
        out.update(rim_underside=z_rim, rim_top=z_rim + ts, eye_underside=z_rim,
                   eye_ring_top=z_rim + ts, blade_top_le=z_rim,
                   total_height=max(z_rim + ts, out["hub_top"]))
    else:
        z_top = tb + params.blade_width
        out.update(blade_top_le=z_top, total_height=max(z_top, out["hub_top"]))
    return out


def shroud_profile(params: ImpellerParams) -> list[tuple[float, float]]:
    """Closed (r, z) polygon of the v0.2 shroud (cone + inlet ring), CCW in the XZ plane."""
    tb = float(params.backplate_thickness)
    ts = float(params.shroud_thickness)
    r2, r_eye = float(params.r2), float(params.r_eye)
    cone, ring_h = float(params.shroud_cone_height), float(params.shroud_inlet_height)
    z_rim = tb + params.b2
    z_ct = z_rim + cone                                  # cone-top underside (at r_eye)
    span = max(r2 - r_eye, 1e-6)
    # outer surface = underside shifted up by ts (axial offset: exact at the flat rim,
    # thickness ts*cos(cone angle) on the cone -- "normal-ish", per SPEC2)
    r_in = r_eye + ts
    z_up_at_r_in = z_ct - cone * ts / span + ts
    pts: list[tuple[float, float]] = [(r_eye, z_ct), (r2, z_rim), (r2, z_rim + ts)]
    if ring_h > ts + 1e-6 and r_in < r2:
        # cone outer surface up to the ring, then the ring to its top, back to the axis side
        pts += [(r_in, z_up_at_r_in), (r_in, z_ct + ring_h), (r_eye, z_ct + ring_h)]
    else:
        # no (or negligible) inlet ring: plain thick cone
        pts += [(r_eye, z_ct + ts)]
    return pts


def blade_top_z(params: ImpellerParams, r: np.ndarray | float) -> np.ndarray | float:
    """Absolute z of the finished (trimmed) blade top edge at radius r, shrouded mode.

    * under the shroud (r >= r_eye): the shroud underside (+ a small overlap so the union
      fuses).  This equals ``min(taper, underside)`` whenever the taper reaches the shroud
      (b1 >= b2 + cone, e.g. the factory default); if the taper would leave a gap under the
      shroud the blade is raised to the underside instead, because a closed impeller's blades
      always span the full passage (a floating shroud would not be one solid).
    * inside the eye (r1 <= r < r_eye): the linear taper tb + h(r).
    """
    tb = float(params.backplate_thickness)
    eps = min(SHROUD_OVERLAP_MM, float(params.shroud_thickness) / 3.0)
    r_arr = np.asarray(r, dtype=float)
    taper = tb + np.asarray(blade_height(params, r_arr))
    under = np.asarray(shroud_underside_z(params, r_arr)) + eps
    z = np.where(r_arr >= params.r_eye, under, taper)
    return z if np.ndim(r) else float(z)


def blade_envelope_profile(params: ImpellerParams) -> list[tuple[float, float]]:
    """(r, z) polygon under which the blades live (top edge = ``blade_top_z``), CCW."""
    tb = float(params.backplate_thickness)
    r1, r2, r_eye = float(params.r1), float(params.r2), float(params.r_eye)
    r_out = r2 + BACKPLATE_RIM_MM + 2.0
    r_in = 0.5 * float(params.r_hub)                  # blades never reach r < r_hub
    z0 = tb - ROOT_OVERLAP_MM
    z_rim = float(blade_top_z(params, r2))
    pts: list[tuple[float, float]] = [(r_in, z0), (r_out, z0), (r_out, z_rim), (r2, z_rim)]
    if r1 < r_eye:
        # under the shroud down to the eye, then a step onto the taper inside the eye
        z_under, z_taper = float(blade_top_z(params, r_eye)), tb + float(blade_height(params, r_eye))
        pts += [(r_eye, z_under)]
        if abs(z_taper - z_under) > 1e-6:
            pts += [(r_eye, z_taper)]
        pts += [(r1, tb + float(blade_height(params, r1)))]
        z_le = tb + float(blade_height(params, r1))
    else:
        pts += [(r1, float(blade_top_z(params, r1)))]
        z_le = float(blade_top_z(params, r1))
    pts += [(r_in, z_le)]
    return pts


def _revolve(profile: list[tuple[float, float]]) -> cq.Workplane:
    """Revolve a closed (r, z) polygon about the global z axis."""
    pts = [(float(r), float(z)) for r, z in profile]
    return cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0))


# ---------------------------------------------------------------------------
# Blades
# ---------------------------------------------------------------------------

def blade_polygon(params: ImpellerParams, i: int) -> np.ndarray:
    """(N, 2) closed polygon of blade *i* in the XY plane (mm)."""
    r, theta = geometry.camber_line(
        params.r1, params.r2, params.blade_angle_inlet, params.blade_angle_outlet,
        n=CAMBER_POINTS,
    )
    theta0 = 2.0 * math.pi * i / params.blade_count
    return geometry.offset_camber(r, theta, params.blade_thickness, theta0)


def _blade_prisms(params: ImpellerParams, z0: float, height: float) -> cq.Workplane:
    """All Z blades extruded from z0 by *height*, as one multi-solid workplane.

    Blades never touch each other (they are separated by the pitch), so a
    single ``extrude`` of all wires produces Z disjoint solids in one shot,
    which is much cheaper than Z separate boolean unions.
    """
    wp = cq.Workplane("XY")
    for i in range(params.blade_count):
        pts = [(float(x), float(y)) for x, y in blade_polygon(params, i)]
        wp = wp.polyline(pts).close()
    return wp.extrude(float(height)).translate((0.0, 0.0, float(z0)))


def _blades(params: ImpellerParams) -> cq.Workplane:
    """v0.1 blades: constant height blade_width, z in [tb, tb + blade_width]."""
    return _blade_prisms(params, params.backplate_thickness, params.blade_width)


def _blades_shrouded(params: ImpellerParams) -> cq.Workplane:
    """v0.2 blades: tapered b1 -> b2 and trimmed by the shroud underside."""
    tb = float(params.backplate_thickness)
    z0 = tb - ROOT_OVERLAP_MM
    prisms = _blade_prisms(params, z0, params.b1 + ROOT_OVERLAP_MM + 1.0)
    envelope = _revolve(blade_envelope_profile(params))
    return prisms.intersect(envelope)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _check_single_valid(body: cq.Workplane, what: str = "impeller") -> cq.Workplane:
    solids = body.solids().vals()
    if len(solids) != 1:
        raise RuntimeError(f"{what} did not fuse into one solid ({len(solids)} solids)")
    if not body.val().isValid():
        raise RuntimeError(f"{what} solid is not valid")
    return body


def _build_open(params: ImpellerParams) -> cq.Workplane:
    """v0.1 geometry (flat optional shroud ring)."""
    tb = float(params.backplate_thickness)
    backplate = cq.Workplane("XY").circle(params.r2 + BACKPLATE_RIM_MM).extrude(tb)
    hub = cq.Workplane("XY").circle(params.r_hub).extrude(_hub_height(params))
    blades = _blades(params)
    body = backplate.union(hub).union(blades)
    if params.shroud:
        shroud = (
            cq.Workplane("XY")
            .workplane(offset=tb + float(params.blade_width))
            .circle(params.r2).circle(params.r1)
            .extrude(float(params.shroud_thickness))
        )
        body = body.union(shroud)
    return body


def _build_shrouded(params: ImpellerParams) -> cq.Workplane:
    """v0.2 geometry: cone + inlet-ring shroud, tapered blades, bore."""
    tb = float(params.backplate_thickness)
    backplate = cq.Workplane("XY").circle(params.r2 + BACKPLATE_RIM_MM).extrude(tb)
    hub = cq.Workplane("XY").circle(params.r_hub).extrude(_hub_height(params))
    blades = _blades_shrouded(params)
    shroud = _revolve(shroud_profile(params))
    body = backplate.union(hub).union(blades).union(shroud)
    if params.hub_bore_diameter > 0:
        h = key_heights(params)["total_height"]
        bore = (
            cq.Workplane("XY").workplane(offset=-1.0)
            .circle(params.hub_bore_diameter / 2.0).extrude(h + 2.0)
        )
        body = body.cut(bore)
    return body


def build_impeller(params: ImpellerParams) -> cq.Workplane:
    """Build the impeller; returns a Workplane holding exactly one valid solid."""
    params.validate()
    body = _build_shrouded(params) if _is_shrouded_v2(params) else _build_open(params)
    return _check_single_valid(body)


def volume(params: ImpellerParams) -> float:
    """Volume [mm^3] of the built solid."""
    return float(build_impeller(params).val().Volume())


def analytic_volume(params: ImpellerParams) -> float:
    """Rough closed-form estimate: backplate + hub + Z * blade_len * t * b (+ shroud, - bore).

    (Ignores the hub/backplate overlap and the blade-end offsets.  In shrouded mode the blade
    volume uses the mean trimmed height along the camber line and the shroud is integrated
    as a revolved profile.)
    """
    r, theta = geometry.camber_line(
        params.r1, params.r2, params.blade_angle_inlet, params.blade_angle_outlet, n=256
    )
    x, y = geometry.polar_to_xy(r, theta)
    seg = np.hypot(np.diff(x), np.diff(y))
    blade_len = float(np.sum(seg))
    v_plate = math.pi * (params.r2 + BACKPLATE_RIM_MM) ** 2 * params.backplate_thickness
    v_hub = math.pi * params.r_hub ** 2 * _hub_height(params)
    if _is_shrouded_v2(params):
        rm = 0.5 * (r[1:] + r[:-1])
        h = np.asarray(blade_top_z(params, rm)) - params.backplate_thickness
        v_blades = params.blade_count * params.blade_thickness * float(np.sum(seg * h))
        v = v_plate + v_hub + v_blades + _polygon_revolved_volume(shroud_profile(params))
        v -= math.pi * (params.hub_bore_diameter / 2.0) ** 2 * _hub_height(params)
        return v
    v_blades = params.blade_count * blade_len * params.blade_thickness * params.blade_width
    v = v_plate + v_hub + v_blades
    if params.shroud:
        v += math.pi * (params.r2 ** 2 - params.r1 ** 2) * params.shroud_thickness
    return v


def _polygon_revolved_volume(profile: list[tuple[float, float]]) -> float:
    """Volume of a closed (r, z) polygon revolved about the z axis (Pappus: 2*pi*A*r_centroid)."""
    p = np.asarray(profile, float)
    r, z = p[:, 0], p[:, 1]
    r_n, z_n = np.roll(r, -1), np.roll(z, -1)
    cross = r * z_n - r_n * z
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-12:
        return 0.0
    r_c = np.sum((r + r_n) * cross) / (6.0 * area)
    return float(2.0 * math.pi * abs(area) * abs(r_c))


# ---------------------------------------------------------------------------
# Meshing / export
# ---------------------------------------------------------------------------

def weld_mesh(V: np.ndarray, F: np.ndarray, decimals: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (OCC tessellates each face separately) and drop degenerate
    triangles.  The result of a valid closed solid is watertight (every edge used twice)."""
    key = np.round(np.asarray(V, float), decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    F2 = inv[np.asarray(F, np.int64)]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    return uniq.astype(np.float64), F2[ok]


def mesh_from_solid(wp: cq.Workplane | cq.Shape, tolerance: float = 0.5,
                    angular_tolerance: float = 0.1, weld: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a solid -> (V: (n,3) float64, F: (m,3) int64).

    The cached OCC triangulation is cleared first so *tolerance* is always honoured.
    Triangle winding is as OCC produces it (outward normals for a valid solid); consumers
    that need two-sided shading should not rely on it.  ``weld`` merges duplicated
    boundary vertices so the mesh is watertight.
    """
    from OCP.BRepTools import BRepTools

    shape = wp.val() if isinstance(wp, cq.Workplane) else wp
    BRepTools.Clean_s(shape.wrapped)
    verts, tris = shape.tessellate(float(tolerance), float(angular_tolerance))
    V = np.asarray([v.toTuple() for v in verts], dtype=np.float64).reshape(-1, 3)
    F = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return weld_mesh(V, F) if weld else (V, F)


def mesh_from_params(params: ImpellerParams, tolerance: float = 0.5,
                     angular_tolerance: float = 0.1, weld: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Build the impeller and tessellate it (see ``mesh_from_solid``)."""
    return mesh_from_solid(build_impeller(params), tolerance, angular_tolerance, weld)


def mesh_face_ids(V: np.ndarray, F: np.ndarray, params: ImpellerParams) -> np.ndarray:
    """Best-effort per-triangle label from the triangle centroid position:

    0 = backplate, 1 = hub, 2 = blades, 3 = shroud.  (Geometric classification of the
    tessellated solid; useful for id-mask rendering.)
    """
    c = V[F].mean(axis=1)
    r = np.hypot(c[:, 0], c[:, 1])
    z = c[:, 2]
    tb = params.backplate_thickness
    ids = np.full(len(F), 2, dtype=np.int64)
    ids[z <= tb + 1e-6] = 0
    ids[(r <= params.r_hub + 1e-6) & (z <= _hub_height(params) + 1e-6)] = 1
    if params.shroud:
        if _is_shrouded_v2(params):
            under = np.asarray(shroud_underside_z(params, r))
            on_shroud = (r >= params.r_eye - 1e-6) & (z >= under - 1e-6)
        else:
            on_shroud = (r >= params.r1 - 1e-6) & (z >= tb + params.blade_width - 1e-6)
        ids[on_shroud] = 3
    return ids


def save_stl(V: np.ndarray, F: np.ndarray, path: str | Path, name: str = "fanscan") -> Path:
    """Write a (V, F) mesh as a binary STL and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)
    tri = V[F]                                            # (m, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 0, n / np.maximum(norm, 1e-30), 0.0).astype(np.float32)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    header = name.encode()[:80].ljust(80, b"\0")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(F)))
        fh.write(rec.tobytes())
    return path


def _export(params: ImpellerParams, path: str | Path, kind: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build_impeller(params), str(path), exportType=kind)
    return path


def export_step(params: ImpellerParams, path: str | Path) -> Path:
    """Write a STEP (AP214) file and return its path."""
    return _export(params, path, "STEP")


def export_stl(params: ImpellerParams, path: str | Path) -> Path:
    """Write an STL mesh (for quick viewing) and return its path."""
    return _export(params, path, "STL")


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from .params import SHROUDED_PARAMS_FACTORY

    ap = argparse.ArgumentParser(description="Build an impeller STEP/STL from a params JSON")
    ap.add_argument("params", nargs="?", help="params JSON (default: built-in DEFAULT_PARAMS)")
    ap.add_argument("--shrouded", action="store_true",
                    help="use SHROUDED_PARAMS_FACTORY() instead of DEFAULT_PARAMS")
    ap.add_argument("--out", default="impeller.step", help="STEP output path")
    ap.add_argument("--stl", default=None, help="optional STL output path")
    ap.add_argument("--tol", type=float, default=0.5, help="tessellation tolerance for --stl")
    a = ap.parse_args(argv)

    if a.params:
        params = ImpellerParams.load(a.params)
    else:
        params = SHROUDED_PARAMS_FACTORY() if a.shrouded else DEFAULT_PARAMS
    t0 = time.perf_counter()
    wp = build_impeller(params)
    t_build = time.perf_counter() - t0
    print(f"built in {t_build:.2f} s, valid={wp.val().isValid()}, "
          f"volume={wp.val().Volume():.0f} mm^3 (analytic ~{analytic_volume(params):.0f})")
    print("STEP ->", export_step(params, a.out))
    if a.stl:
        V, F = mesh_from_solid(wp, a.tol)
        print(f"STL  -> {save_stl(V, F, a.stl)}  ({len(V)} vertices, {len(F)} triangles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
