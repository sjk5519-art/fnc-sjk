"""CAD model of a centrifugal impeller (cadquery) from ``ImpellerParams``.

Geometry (z up, backplate on the bottom):

* backplate: cylinder r = r2 + 2 mm, z in [0, tb]
* hub:       cylinder r = r_hub, from z = 0, height = hub_height or (blade_width + tb)
* blades:    Z extruded ``geometry.offset_camber`` polygons, z in [tb, tb + blade_width],
             blade i rotated by 2*pi*i/Z (camber line sweeps toward +theta, CCW, as r grows)
* shroud:    optional ring r1..r2 of ``shroud_thickness`` sitting on top of the blades

Everything is unioned into one solid.  All lengths are millimetres.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cadquery as cq
import numpy as np

from . import geometry
from .params import DEFAULT_PARAMS, ImpellerParams

BACKPLATE_RIM_MM = 2.0      # backplate radius = r2 + this
CAMBER_POINTS = 64          # >= 48 per SPEC so the blade polygon is smooth


def _hub_height(params: ImpellerParams) -> float:
    if params.hub_height is not None:
        return float(params.hub_height)
    return float(params.blade_width + params.backplate_thickness)


def blade_polygon(params: ImpellerParams, i: int) -> np.ndarray:
    """(N, 2) closed polygon of blade *i* in the XY plane (mm)."""
    r, theta = geometry.camber_line(
        params.r1, params.r2, params.blade_angle_inlet, params.blade_angle_outlet,
        n=CAMBER_POINTS,
    )
    theta0 = 2.0 * math.pi * i / params.blade_count
    return geometry.offset_camber(r, theta, params.blade_thickness, theta0)


def _blades(params: ImpellerParams) -> cq.Workplane:
    """All Z blades as one solid, extruded from one multi-wire workplane.

    Blades never touch each other (they are separated by the pitch), so a
    single ``extrude`` of all wires produces Z disjoint solids in one shot,
    which is much cheaper than Z separate boolean unions.
    """
    wp = cq.Workplane("XY")
    for i in range(params.blade_count):
        pts = [(float(x), float(y)) for x, y in blade_polygon(params, i)]
        wp = wp.polyline(pts).close()
    return wp.extrude(float(params.blade_width)).translate(
        (0.0, 0.0, float(params.backplate_thickness))
    )


def build_impeller(params: ImpellerParams) -> cq.Workplane:
    """Build the impeller; returns a Workplane holding exactly one valid solid."""
    params.validate()
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

    solids = body.solids().vals()
    if len(solids) != 1:
        raise RuntimeError(f"impeller did not fuse into one solid ({len(solids)} solids)")
    if not body.val().isValid():
        raise RuntimeError("impeller solid is not valid")
    return body


def volume(params: ImpellerParams) -> float:
    """Volume [mm^3] of the built solid."""
    return float(build_impeller(params).val().Volume())


def analytic_volume(params: ImpellerParams) -> float:
    """Rough closed-form estimate: backplate + hub + Z * blade_len * t * b.

    (Ignores the hub/backplate overlap and the blade-end offsets.)
    """
    r, theta = geometry.camber_line(
        params.r1, params.r2, params.blade_angle_inlet, params.blade_angle_outlet, n=256
    )
    x, y = geometry.polar_to_xy(r, theta)
    blade_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    v_plate = math.pi * (params.r2 + BACKPLATE_RIM_MM) ** 2 * params.backplate_thickness
    v_hub = math.pi * params.r_hub ** 2 * _hub_height(params)
    v_blades = params.blade_count * blade_len * params.blade_thickness * params.blade_width
    v = v_plate + v_hub + v_blades
    if params.shroud:
        v += math.pi * (params.r2 ** 2 - params.r1 ** 2) * params.shroud_thickness
    return v


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

    ap = argparse.ArgumentParser(description="Build an impeller STEP/STL from a params JSON")
    ap.add_argument("params", nargs="?", help="params JSON (default: built-in DEFAULT_PARAMS)")
    ap.add_argument("--out", default="impeller.step", help="STEP output path")
    ap.add_argument("--stl", default=None, help="optional STL output path")
    a = ap.parse_args(argv)

    params = ImpellerParams.load(a.params) if a.params else DEFAULT_PARAMS
    t0 = time.perf_counter()
    wp = build_impeller(params)
    t_build = time.perf_counter() - t0
    print(f"built in {t_build:.2f} s, valid={wp.val().isValid()}, "
          f"volume={wp.val().Volume():.0f} mm^3 (analytic ~{analytic_volume(params):.0f})")
    print("STEP ->", export_step(params, a.out))
    if a.stl:
        print("STL  ->", export_stl(params, a.stl))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
