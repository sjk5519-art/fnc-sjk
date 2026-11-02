"""Synthetic axial-view photographs of a centrifugal impeller.

The renderer draws the impeller described by an :class:`ImpellerParams` the way
a camera looking along the axis would see it (inlet eye toward the camera, no
shroud), using the intensity levels of the shared *Image convention*:

    background  ~ 30     backplate disc (r <= r2 + rim)  ~ 200
    blades      ~ 70     hub (r <= r_hub)                ~ 110
    reference circle (known physical diameter, top-left corner)  255

Coordinate convention
---------------------
The blade geometry (:mod:`fanscan.geometry`) lives in a y-UP math frame where
the blade sweeps toward +theta (counter-clockwise) as r grows.  OpenCV images
are y-DOWN, so a math point ``(x, y)`` [mm] is drawn at

    px = cx + x * s          py = cy - y * s          (s = px_per_mm)

Consequently the +theta sweep appears *clockwise* on screen.  ``rotation_deg``
is applied in the math frame (positive = CCW in the y-up frame = clockwise on
screen).  The analyzer must undo exactly this mapping when unwrapping.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import camber_line, offset_camber
from .params import DEFAULT_PARAMS, ImpellerParams

# ---- intensity levels (8-bit) --------------------------------------------
BACKGROUND_LEVEL = 30
BACKPLATE_LEVEL = 200
BLADE_LEVEL = 70
HUB_LEVEL = 110
REF_LEVEL = 255

# ---- layout constants ------------------------------------------------------
RIM_MARGIN_MM = 2.0        # backplate radius = r2 + RIM_MARGIN_MM (matches cad.py)
_REF_PAD_PX = 12           # gap between the image border and the reference circle
_REF_GAP_PX = 10           # minimum clearance between reference circle and backplate
_CAMBER_SAMPLES = 96       # samples along the camber line per blade polygon
_FIXED_POINT_SHIFT = 6     # sub-pixel precision bits for cv2.fillPoly / cv2.circle
# OpenCV fills are boundary-inclusive (plus an anti-aliasing fringe): a polygon
# of geometric width W covers ~W + 1.4 px, a circle of radius R has an
# intensity-weighted radius of ~R + 0.6 px.  Shrink the shapes by this much so
# the drawn footprint matches the geometry the truth dict reports.
_POLY_EDGE_PX = 0.7        # per side, for cv2.fillPoly
_CIRCLE_EDGE_PX = 0.5      # radius, for cv2.circle
_MIN_DRAWN_WIDTH_PX = 0.5  # never shrink a blade polygon below this width


def _to_px(x_mm: np.ndarray, y_mm: np.ndarray, cx: float, cy: float, s: float
           ) -> np.ndarray:
    """Math frame (y-up, mm) -> image frame (y-down, px) as (N, 2) float array."""
    return np.stack([cx + x_mm * s, cy - y_mm * s], axis=1)


def _fixed(v: float | np.ndarray) -> Any:
    """Scale to fixed-point integers for the OpenCV ``shift`` argument."""
    return np.round(np.asarray(v, dtype=float) * (1 << _FIXED_POINT_SHIFT)).astype(np.int32)


def _layout(params: ImpellerParams, px_per_mm: float, ref_diameter_mm: float,
            center_jitter_px: tuple[float, float]) -> dict[str, Any]:
    """Image size and placement of impeller + reference circle."""
    s = px_per_mm
    disc_r_px = (params.r2 + RIM_MARGIN_MM) * s
    ref_r_px = 0.5 * ref_diameter_mm * s
    jit = math.hypot(*center_jitter_px)

    # The reference circle sits at (pad + rr, pad + rr).  With the impeller at
    # the image centre (W/2, W/2) and W/2 = disc_r + m, the corner distance is
    # sqrt(2) * (disc_r + m - pad - rr) and must exceed disc_r + rr + gap (+jitter).
    need = (disc_r_px + ref_r_px + _REF_GAP_PX + jit) / math.sqrt(2.0) \
        - disc_r_px + _REF_PAD_PX + ref_r_px
    margin = max(need, 0.04 * disc_r_px + jit + 4.0)
    half = disc_r_px + margin
    size = int(math.ceil(2.0 * half))
    size += size % 2  # even size so the nominal centre is on a half-pixel grid

    cx = size / 2.0 + center_jitter_px[0]
    cy = size / 2.0 + center_jitter_px[1]
    ref_c = (_REF_PAD_PX + ref_r_px, _REF_PAD_PX + ref_r_px)
    return {
        "size": size, "cx": cx, "cy": cy, "disc_r_px": disc_r_px,
        "ref_r_px": ref_r_px, "ref_c": ref_c,
    }


def render(params: ImpellerParams, px_per_mm: float = 4.0, ref_diameter_mm: float = 30.0,
           noise_sigma: float = 0.0, blur_sigma: float = 0.0, rotation_deg: float = 0.0,
           center_jitter_px: tuple[float, float] = (0, 0), seed: int = 0
           ) -> tuple[np.ndarray, dict]:
    """Render a synthetic axial photo.

    Returns ``(img, truth)`` where *img* is an 8-bit grayscale image and *truth*
    holds the ground truth needed to score an analysis: ``px_per_mm``,
    ``center_xy``, ``ref_center_xy``, ``ref_diameter_mm``, ``rotation_deg`` and
    ``params`` (``params.to_dict()``), plus a few extra render details.
    """
    params.validate()
    if px_per_mm <= 0 or ref_diameter_mm <= 0:
        raise ValueError("px_per_mm and ref_diameter_mm must be positive")

    s = float(px_per_mm)
    lay = _layout(params, s, ref_diameter_mm, center_jitter_px)
    size, cx, cy = lay["size"], lay["cx"], lay["cy"]
    img = np.full((size, size), BACKGROUND_LEVEL, dtype=np.uint8)

    def disc(c: tuple[float, float], r_px: float, level: int) -> None:
        """Filled circle whose drawn footprint has radius *r_px*."""
        r_draw = max(r_px - _CIRCLE_EDGE_PX, 0.5)
        cv2.circle(img, tuple(int(v) for v in _fixed(list(c))), int(_fixed(r_draw)),
                   level, thickness=-1, lineType=cv2.LINE_AA, shift=_FIXED_POINT_SHIFT)

    # Backplate disc (light), r <= r2 + rim margin.
    disc((cx, cy), lay["disc_r_px"], BACKPLATE_LEVEL)

    # Blades (dark), one camber line rotated Z times.  Polygons are built in mm
    # in the y-up math frame and mapped to the y-down image frame.  The polygon
    # is offset by the fill's boundary excess so the drawn width equals
    # blade_thickness (see _POLY_EDGE_PX).
    r_mm, theta = camber_line(params.r1, params.r2, params.blade_angle_inlet,
                              params.blade_angle_outlet, n=_CAMBER_SAMPLES)
    t_px = params.blade_thickness * s
    t_draw_mm = max(t_px - 2.0 * _POLY_EDGE_PX, min(t_px, _MIN_DRAWN_WIDTH_PX)) / s
    rot = math.radians(rotation_deg)
    Z = params.blade_count
    for i in range(Z):
        poly_mm = offset_camber(r_mm, theta, t_draw_mm,
                                theta0=2.0 * math.pi * i / Z + rot)
        poly_px = _to_px(poly_mm[:, 0], poly_mm[:, 1], cx, cy, s)
        cv2.fillPoly(img, [_fixed(poly_px).reshape(-1, 1, 2)], BLADE_LEVEL,
                     lineType=cv2.LINE_AA, shift=_FIXED_POINT_SHIFT)

    # Hub boss (mid-dark) on top of everything inside r_hub.
    disc((cx, cy), params.r_hub * s, HUB_LEVEL)

    # Scale reference: solid white circle, top-left, fully outside the impeller.
    rcx, rcy = lay["ref_c"]
    disc((rcx, rcy), lay["ref_r_px"], REF_LEVEL)

    # Degradations: blur first (optics), then sensor noise.
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noisy = img.astype(np.float32) + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)

    truth: dict[str, Any] = {
        "px_per_mm": s,
        "center_xy": (float(cx), float(cy)),
        "ref_center_xy": (float(rcx), float(rcy)),
        "ref_diameter_mm": float(ref_diameter_mm),
        "rotation_deg": float(rotation_deg),
        "params": params.to_dict(),
        # extras
        "image_size": (int(size), int(size)),
        "ref_radius_px": float(lay["ref_r_px"]),
        "backplate_radius_px": float(lay["disc_r_px"]),
        "rim_margin_mm": RIM_MARGIN_MM,
        "r1_px": float(params.r1 * s),
        "r2_px": float(params.r2 * s),
        "hub_radius_px": float(params.r_hub * s),
        "noise_sigma": float(noise_sigma),
        "blur_sigma": float(blur_sigma),
        "center_jitter_px": (float(center_jitter_px[0]), float(center_jitter_px[1])),
        "seed": int(seed),
        "levels": {
            "background": BACKGROUND_LEVEL, "backplate": BACKPLATE_LEVEL,
            "blade": BLADE_LEVEL, "hub": HUB_LEVEL, "reference": REF_LEVEL,
        },
    }
    return img, truth


def save(path: str | Path, img: np.ndarray) -> Path:
    """Write *img* to *path* (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write image to {path}")
    return path


def save_truth(path: str | Path, truth: dict) -> Path:
    """Write the ground-truth dict as JSON next to an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    return path


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m fanscan.synth",
                                 description="Render a synthetic impeller photo + truth JSON.")
    ap.add_argument("out", nargs="?", default="synth_sample.png", help="output PNG path")
    ap.add_argument("--params", help="ImpellerParams JSON (default: DEFAULT_PARAMS)")
    ap.add_argument("--px-per-mm", type=float, default=4.0)
    ap.add_argument("--ref-diameter-mm", type=float, default=30.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    params = ImpellerParams.load(a.params) if a.params else DEFAULT_PARAMS
    img, truth = render(params, px_per_mm=a.px_per_mm, ref_diameter_mm=a.ref_diameter_mm,
                        noise_sigma=a.noise, blur_sigma=a.blur, rotation_deg=a.rotation,
                        seed=a.seed)
    out = save(a.out, img)
    truth_path = save_truth(out.with_suffix(".truth.json"), truth)
    print(f"wrote {out} ({img.shape[1]}x{img.shape[0]}) and {truth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
