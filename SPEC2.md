# fanscan v0.2 — shrouded (production-type) impeller mode

Goal: analyse photographs of a *closed* centrifugal impeller (front shroud with a conical inlet section and an
inlet ring, bolted hub flange with a through-bore) taken under **imperfect** conditions (tilt, perspective, cluttered
background, uneven lighting, no reference object), and still produce `ImpellerParams` incl. **beta2**, each value
tagged with a provenance (`measured` / `extrapolated` / `wrap` / `prior` / `user`) and a confidence in [0, 1].

Real samples (no ground truth, qualitative only): `samples/real_axial_eye.jpg` (through-the-eye view, close-up,
tilted, cluttered desk), `samples/real_side_oblique.jpg` (low oblique view showing the shroud cone, rim gap, blade
trailing edges).  Look at both before designing anything.

Everything from SPEC.md (v0.1) stays valid and must keep passing.  Already written for v0.2 (read them):
- `params.py`: new optional fields `eye_diameter, shroud_cone_height, shroud_inlet_height, hub_bore_diameter,
  blade_width_inlet`; properties `r_eye, b1, b2, is_shrouded`; `SHROUDED_PARAMS_FACTORY()` (300 mm, Z=11, beta 30/40,
  b1/b2 75/55, eye 180, cone 20, inlet ring 18, bore 60).
- `geometry.py`: `wrap_from_betas`, `beta2_from_wrap` (brentq), `wrap_mod_pitch_candidates`.

## Physical model of the shrouded impeller (z up, backplate top face at z = tb)
- Backplate disc r <= r2 + 2, z in [0, tb]; through-bore r < r_bore removed from everything.
- Hub boss r <= r_hub, z in [0, tb + hub_h] where hub_h = hub_height or b1 (a short flange is fine).
- Blades: camber from geometry.camber_line, thickness t, bottom at z = tb, top edge following the **shroud underside**:
  blade height h(r) = b2 + (b1 - b2) * (r2 - r) / (r2 - r1) for r in [r1, r2] (linear taper, b1 >= b2).
- Shroud: a revolved profile. Underside: z = tb + h(r) for r in [r_eye, r2] (so at the rim the gap height is b2, at the eye
  it is b1 + the cone: define the cone as the shroud rising from z = tb + b2 at r2 to z = tb + b2 + shroud_cone_height at
  r_eye — for the default these are consistent with b1 = b2 + cone, i.e. 75 = 55 + 20; if b1 != b2 + cone, the blade is
  trimmed by the shroud underside, which is what matters).  Thickness shroud_thickness normal-ish (use a radial offset
  of the profile; exactness is not important).  Inlet ring: cylinder r in [r_eye, r_eye + shroud_thickness],
  z from the cone top up by shroud_inlet_height.
- Blade leading edge at r1 (r1 <= r_eye typically; default r1 = 95 vs r_eye = 90 — i.e. the LE sits just *outside*
  the eye radius, hidden under the shroud lip; this is realistic and is why the LE is only seen at a grazing angle).
  NOTE: blades are visible through the eye out to a radius that depends on camera distance; see the analyzer section.

## Module A: `fanscan/cad.py` (extend, keep v0.1 API/behaviour for non-shrouded params)
- `build_impeller(params)`: when `params.shroud` and `params.eye_diameter` is set, build the full model above (revolve the
  shroud profile with `cq.Workplane("XZ").polyline(...).close().revolve(360, (0,0,0), (0,1,0))`, blades = extrude to
  z = tb + b1 then `.intersect()` with the under-shroud "air" solid or `.cut()` with the shroud solid; bore via cut).
  Must give ONE valid solid in < 60 s for the default shrouded params.  Keep the v0.1 flat-ring shroud when
  eye_diameter is None.
- `mesh_from_params(params, tolerance=0.5) -> (V: (n,3) float64, F: (m,3) int64)` via `solid.tessellate(tolerance)`,
  plus `mesh_from_solid(wp, tolerance)`.  Also `save_stl`.

## Module B: `fanscan/render3d.py` — dependency-free software renderer for photo-like test images
```
@dataclass
class Camera:
    position: np.ndarray   # (3,) mm, world frame = impeller frame (z up, origin at backplate bottom centre)
    target:   np.ndarray   # (3,) look-at point
    up:       np.ndarray   # (3,)
    focal_px: float        # pinhole focal length in px
    size:     tuple[int, int]   # (width, height)
    def project(self, P: np.ndarray) -> np.ndarray     # (n,3) world -> (n,2) px (+ depth as 3rd column)
    def intrinsics(self) -> np.ndarray                  # 3x3 K
    def extrinsics(self) -> np.ndarray                  # 3x4 [R|t]

def render(V, F, cam: Camera, *, light_dir=(0.3,-0.4,1.0), ambient=0.35, albedo=0.85,
           background: np.ndarray | float = 0.12, noise_sigma=0.0, blur_sigma=0.0, seed=0,
           return_depth=False) -> np.ndarray (uint8 gray HxW) [, depth float32]
def clutter_background(size, seed) -> np.ndarray      # procedural desk-like clutter: light wood tone, dark rectangles,
                                                      # thin dark 'cable' curves, a bright blob; uint8 gray
def camera_preset(name, params, size=(1200,1200), **kw) -> Camera
   names: 'axial_far' (on axis, distance 6*D2), 'axial_close' (distance 1.2*D2, phone-like focal = 0.9*width),
          'tilt' (axis tilt tilt_deg=15, distance 3*D2), 'oblique' (elevation 40 deg, distance 2.5*D2),
          'side' (elevation 5 deg, distance 2.5*D2, aimed at the rim gap)
def render_params(params, preset='axial_close', **kw) -> (img, truth dict)
   truth: camera K/R/t, px_per_mm at the backplate plane on axis, projected centre of the impeller axis, projected rim
   ellipse (cv2.fitEllipse of projected rim circle points) and eye ellipse, `params.to_dict()`, visible blade fraction
   (max r at which blade root pixels are visible through the eye — compute from the depth buffer/ids by rendering a
   blade-id mask: give `render` an optional `face_ids` array and return the id image).
```
Implementation: z-buffer rasterisation.  Loop over triangles in numpy with per-triangle bounding boxes and a
vectorised barycentric inside-test; or group triangles by bbox size.  Target: 20k triangles at 1200x1200 in < 15 s.
Lambert shading with a tiny specular term; faces are two-sided.  Perspective-correct depth is not required.
A 1-px Gaussian blur after shading is fine.  Provide `__main__` that renders all 5 presets of SHROUDED_PARAMS_FACTORY()
to a directory and VIEW the PNGs (Read tool) — they must look like the real samples: light impeller, blades visible
through the eye with shading, darker rim gap with trailing edges visible in 'side'/'oblique'.

## Module C: `fanscan/pose.py` — rim/eye ellipse detection and rectification
```
@dataclass
class EllipseFit: center: (cx, cy); axes: (a, b)  # semi-axes, a >= b; angle_deg; score: float; n_inliers: int
def find_ellipses(img_gray, *, min_axis_frac=0.15, max_count=6) -> list[EllipseFit]
   Robust pipeline: CLAHE -> Canny at 2-3 thresholds -> contours -> fitEllipse for long contours -> RANSAC-style
   support scoring (fraction of the ellipse perimeter with an edge pixel within 2 px) -> non-max suppression of near-
   duplicates.  Must find the outer rim and the inlet-ring ellipses in `samples/real_axial_eye.jpg` and in render3d
   'tilt'/'axial_close' images despite clutter.
def pick_impeller_ellipses(ellipses, img_shape) -> (rim: EllipseFit, eye: EllipseFit | None)
   rim = large, high-score, roughly concentric family; eye = smaller ellipse with similar orientation & aspect inside it.
def rectify(img, rim: EllipseFit, out_size=None) -> (img_rect, M: 2x3 affine, px_per_unit)
   Affine map that sends the rim ellipse to a circle of radius a centred in the output (undoes tilt to first order).
def tilt_from_ellipse(e) -> float   # degrees, acos(b/a)
def estimate_perspective_scale(rim, eye, rim_z, eye_z) -> float | None   # optional: camera distance along the axis from
   the two ellipses if their physical radii ratio & heights are known; return None if ill-conditioned.
```

## Module D: `fanscan/shrouded.py` — the shrouded-mode analyzer
```
@dataclass
class Provenance: value: float; source: str  # 'measured'|'extrapolated'|'wrap'|'prior'|'user'; confidence: float; note: str
@dataclass
class ShroudedResult: params: ImpellerParams; provenance: dict[str, Provenance]; rim: EllipseFit; eye: EllipseFit|None;
                      rectified: np.ndarray; debug: dict
def analyze_shrouded(img, *, outer_diameter_mm=None, eye_diameter_mm=None, ref_diameter_mm=None,
                     blade_width_mm=None, blade_width_inlet_mm=None, blade_count_hint=None,
                     rim_images: list[np.ndarray] | None = None, marker_hint=None) -> ShroudedResult
def draw_overlay(img, result) -> np.ndarray
```
Algorithm:
1. `pose.find_ellipses` -> rim & eye.  Rectify on the rim ellipse.  Work in the rectified image from here; everything
   in px of the rectified frame, centre = rim centre.
2. **Scale** (priority): `ref_diameter_mm` (white disc, as v0.1) > `outer_diameter_mm` (rim) > `eye_diameter_mm` (eye
   ellipse) > **prior**: no scale -> set D2 := 300 mm nominal with provenance 'prior', confidence 0.05, and report all
   diameters consistently (ratios are still measured).  Note the rim and eye are at different heights: with a close
   camera the eye appears larger relative to the rim than it is.  If both D2 and D_eye are given, use
   `pose.estimate_perspective_scale` to get the camera distance and correct the *backplate-plane* scale used for blade
   tracing (the blade roots are at z ~ tb, the rim at z ~ tb + b2, the eye ring at tb + b1 + ring).  If only one is
   given, assume a far camera and add a warning.
3. **Eye region**: mask r < r_eye_px (rectified).  Inside it: hub flange + bore (dark hole, maybe clutter through it)
   at the centre, blades between r_hub and the eye edge.  Hub radius: radial profile of edge density/intensity; bore:
   dark central blob.  If not found, prior hub = 0.3*D2 ('prior').
4. **Blade count Z**: polar unwrap the annulus r in [r_hub+margin, r_eye-margin]; use the angular *gradient* profile
   (|d/dtheta| of a CLAHE-normalised image, summed over radii) — blades appear as edge pairs regardless of albedo;
   harmonic-sum FFT like v0.1; majority vote over radii.  Cross-check with the real sample (expect 10..12 — count them
   by eye first and record the number you see in debug['expected_from_eye']).  blade_count_hint breaks ties.
5. **Blade root tracing**: in the polar gradient image, for each radius row find the Z strongest edge positions, link
   across radii (unwrap 2pi).  A blade produces two edges (root crease on one side, top/TE highlight on the other);
   choose the edge family with the most consistent, smooth theta(r), or trace the mid-line of each edge pair.  Trace
   from r_min (just outside hub) to r_vis (where the edge support dies out under the shroud).  Fit geometry.fit_linear_
   beta on r in mm with r1 := the smallest traced radius (LE is at r1; if the LE is not visible, r1 := r_eye with
   provenance 'prior').  -> beta1 'measured' (confidence from fit residual), beta2_extrap 'extrapolated' with
   confidence = clip((r_vis - r1)/(r2 - r1), 0, 1) * fit quality.  Report visible fraction in debug.
6. **Wrap-angle beta2** (if `rim_images` given): each rim image is a 'side' view of the rim gap.  Find the rim (top edge
   of the gap = shroud lower edge, bottom = backplate top edge: two nearly parallel bright/dark transitions), the trailing
   edges (vertical dark/bright lines across the gap), and the marker (a saturated/bright or very dark small blob on the
   shroud rim; `marker_hint` = 'bright'|'dark').  TE azimuths relative to the marker: phi = asin((x - x_marker)/r2_px)
   for points near the image centre (use only |x - x_c| < 0.6 r2_px, i.e. +-37 deg; take several images around the rim
   if given).  The LE azimuths relative to the same marker come from the axial image (marker must be visible there on
   the shroud rim).  wrap_mod = median over blades of (phi_TE - phi_LE) mod pitch; candidates from
   geometry.wrap_mod_pitch_candidates with est = the extrapolated wrap; beta2 = geometry.beta2_from_wrap -> 'wrap',
   confidence 0.8 if the azimuth residuals are small.  Without rim images, skip and keep 'extrapolated'.
7. **b (blade width)**: user value > from a rim image (gap height / rim diameter ratio * D2) > prior 0.2*D2 ('prior').
   b1: user > b2 + cone estimate (cone height not measurable axially -> prior b1 = 1.3*b2).
8. Assemble params (validate; clamp into valid ranges with warnings), provenance for every field, debug.
Overlay: rim/eye ellipses on the ORIGINAL image (inverse-map), hub circle, blade traces + extrapolated camber to r2
(dashed), marker, text with Z, betas and their provenance.

## Module E: `fanscan/cli.py` (extend) + tests
- `analyze --mode {open,shrouded}` (default: auto -> shrouded if an eye ellipse is found inside the rim and the
  v0.1 reference disc is absent) with the new options `--eye-diameter-mm, --blade-width-inlet-mm, --rim-image (repeatable),
  --marker {bright,dark}`.  Print the provenance table.
- `render --preset NAME --out img.png [--params p.json] [--clutter] [--noise N] [--blur B] [--tilt-deg T]`
- `pipeline3d --out-dir DIR --preset NAME [...]`: render -> analyze_shrouded -> build -> comparison table with the same
  pass/fail rules as v0.1 but beta2 is 'info' unless rim images were used.
- tests: `test_render3d.py` (renders all presets; projected rim ellipse in truth matches a cv2.fitEllipse of rendered
  rim edge within 2 px; render time bound), `test_pose.py` (synthetic ellipses on clutter -> found within 1 px / 1 deg;
  real sample -> rim & eye found), `test_shrouded.py` grid: presets {axial_far, axial_close, tilt(10), tilt(20)} x
  Z {9, 11, 13} x beta2 {30, 40, 55} x clutter {off, on}: Z exact, D2/D_eye/hub within 4 % when D2 given, beta1 within
  5 deg, extrapolated beta2 within 8 deg for axial_close (document the actual achieved numbers in the test file header
  rather than inventing tolerances — loosen with a comment if needed but never silently skip), plus a wrap-angle test:
  render 'side' at 3 azimuths with a marker (add a small bright sphere/box on the shroud rim to the mesh in
  render_params when `marker=True`) and check beta2 'wrap' within 3 deg.  Keep the whole suite under ~4 min.
