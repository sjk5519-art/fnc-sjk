# fanscan v0.3 — marker-free beta2 from rim-gap (side) photographs

Constraint from the user: **nothing may be attached to or drawn on the fan.** beta2 must come from camera angles alone.
The wrap-angle path (marker in two photos) stays available but is no longer the primary route.

## Principle ("edge-on transition")
Look at the rim gap (between shroud and backplate) from the side. Next to each blade trailing edge (TE) the blade's
pressure or suction face is visible as a shaded strip. For a rim point at azimuth phi (math frame, +x axis = 0, CCW),
tangent t = (-sin phi, cos phi), radial r = (cos phi, sin phi), the blade exit direction for the CCW-sweeping blade is
e = cos(beta2) t + sin(beta2) r = (sin(beta2 - phi), cos(beta2 - phi)) and the face normal n = (-cos(beta2-phi),
sin(beta2-phi)).  A far camera on the -y axis looking along v = (0, cos eps, -sin eps) (elevation eps) sees
v . n = cos(eps) sin(beta2 - phi): the face is **edge-on** where phi0 = beta2 - 180 deg on the near side, i.e. at image
offset x0 = -r2 cos(beta2) from the rim centre for CCW blades (+r2 cos(beta2) for CW blades).  So:
    |x0| / a_rim = cos(beta2)        (a_rim = rim semi-major axis in px; elevation does not move x0 for 2-D blades)
and the side of the centre on which the transition lies gives the handedness.  Verified on render3d 'side' renders:
beta2 = 30 -> |x0| ~ 212 px of a = 244 px (cos 30 = 0.866 -> 211 px); beta2 = 55 -> ~140 px (cos 55 = 0.574 -> 140 px).
Perspective (close camera) shifts x0 slightly: use the true ray direction per TE when the camera distance is known
(D2 + focal_px / EXIF) and otherwise assume a far camera and widen the confidence interval.

Between the TEs the transition must be interpolated: measure for each visible TE_i the SIGNED apparent width w_i of the
face strip adjacent to it (positive = strip on the +x side of the TE line). w_i ~ k * sin(beta2 - phi_i) * d_i where d_i
is the visible depth to the occluder; the sign flips at phi0. With Z = 8 only ~4-5 TEs are on the near half, so the
bracket is one pitch (45 deg) wide: fit the model to all w_i (not just the two bracketing ones), and combine SEVERAL side
photos taken from different azimuths (each photo gives an independent phi0; no registration between photos is needed).

## Refinement by synthesis (optional stage, strongly recommended)
Build a cheap blade-only mesh WITHOUT CadQuery (numpy: each blade = camber polyline offset by t/2, extruded between
z = tb and the shroud underside; plus backplate and shroud discs/cone as triangle fans) in `fanscan/fastmesh.py`,
render the rim-gap strip with render3d.render for candidate beta2 in a grid, and maximise the similarity between the
rendered and the photographed gap strip (normalised cross-correlation of the horizontal gradient / TE-face pattern after
rectifying both strips to the same (x, z) grid). This handles perspective, elevation, blade curvature and occlusion
exactly. Mesh build must be < 50 ms and a 1400x400 gap render < 1.5 s so that a 15..75 deg sweep in 2 deg steps plus a
fine 0.5 deg pass fits in ~60 s.

## Module: `fanscan/rimgap.py`
```
@dataclass
class RimGapResult:
    beta2: float; confidence: float; handedness: str   # 'ccw' | 'cw' | 'undetermined'
    x0_px: float | None            # fitted edge-on offset from the rim centre (image px)
    te_x: list[float]; te_widths: list[float]   # detected TE positions and signed face widths
    rim: pose.EllipseFit | None; gap: dict      # gap top/bottom lines, px
    per_image: list[dict]          # when several images are given
    method: str                    # 'sign-change' | 'synthesis' | 'prior'
    debug: dict; warnings: list[str]
def analyze_rim_gap(images: list[np.ndarray], *, blade_count: int, outer_diameter_mm=None, focal_px=None,
                    beta1_hint=None, blade_width_mm=None, refine=True) -> RimGapResult
def draw_rim_gap_overlay(img, result, index=0) -> np.ndarray
```
Steps per image: rim ellipse (pose.find_ellipses; in a side view the rim is a very flat ellipse — the *backplate* rim and
the *shroud* rim are two nearly parallel arcs; take the backplate one for a, centre x) -> gap band (reuse/adapt the gap
and TE detection in shrouded.analyze_rim_image) -> per-TE signed face width from the intensity structure left/right of
the TE line (face = region with a distinct, roughly uniform shade bounded by the TE line and the next edge; use the
horizontal gradient sign pattern and shading contrast; be robust to the wood-grain clutter background above/below the
gap by working strictly inside the gap band) -> fit phi0 -> beta2 + handedness; optional synthesis refinement.
Combine images: inverse-variance weighted mean of beta2; disagreement > 8 deg -> confidence down + warning.
Confidence: from the fit residual, number of TEs on each side of the transition, image count, and far-camera assumption.

## Integration
- shrouded.analyze_shrouded: `rim_images` without a marker now routes to analyze_rim_gap; result -> beta2 provenance
  'rimgap' (new source), handedness cross-checked; wrap path only when marker_hint is given.
- cli: `analyze --rim-image` works without --marker; new `rimgap` subcommand for side photos alone:
  `python -m fanscan.cli rimgap side1.jpg [side2.jpg ...] --blade-count 8 [--outer-diameter-mm 300] [--overlay dir]`
- render3d: 'side' preset with azimuth/elevation/distance already exists; add `--elevation-deg` coverage 3..35.

## Validation (tests/test_rimgap.py)
Grid on render3d 'side' renders (realistic variant): beta2 {25, 35, 45, 55, 65} x Z {8, 11} x elevation {5, 15, 30} x
azimuth {random 3} x clutter {off, on}: single image |beta2 err| <= 5 deg (sign-change) / <= 3 deg (synthesis),
3 images <= 2 deg; handedness always right; CW-mirrored renders (flip the image horizontally) give 'cw' and the same
beta2. Real sample `samples/real_side_oblique.jpg` (Z = 8, elevation ~15 deg, slight yaw): must not crash, must detect
the gap and >= 3 TEs, report beta2 with a confidence and handedness consistent with the axial analysis ('cw' when seen
from the inlet; CHECK the sign convention carefully — a side view from the outside sees the mirror of the inlet view).
Record the achieved numbers in the test header. Keep the suite < 4 min with the render cache (tests/conftest.py).
