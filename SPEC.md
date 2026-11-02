# fanscan v0.1 — implementation spec

Photo (axial view of a centrifugal impeller) -> `ImpellerParams` -> STEP.

Python 3.13, venv at `.venv` (numpy, opencv-python-headless, scipy, cadquery, pytest).
Run everything with `. .venv/bin/activate`.

## Already written (do NOT change the public API; read them first)
- `fanscan/params.py` — `ImpellerParams` dataclass, `DEFAULT_PARAMS`, `validate()`, `save/load`, `compare()`.
- `fanscan/geometry.py` — `camber_line(r1, r2, beta1, beta2, n) -> (r, theta)`,
  `offset_camber(r, theta, thickness, theta0) -> (N,2) polygon`, `fit_linear_beta(r, theta, r1, r2) -> (b1, b2)`,
  `beta_from_trace`, `polar_to_xy`, `total_sweep_deg`.
  **Blade convention:** theta(r1)=0, blade sweeps toward +theta (CCW in a y-up frame) as r grows.
  All functions are unit-agnostic (mm or px), theta in radians.

## Image convention (shared by synth and analyze)
- 8-bit grayscale (BGR input must be converted). Camera looks along the axis; the inlet eye faces the camera
  so blades are visible between the hub and the outer rim (no shroud in the synthetic images).
- Background: dark (~30). Backplate disc (r <= r2 + rim margin): light (~200). Blades: darker (~70) drawn on
  the backplate. Eye region (r < r1) inside the blades shows the backplate too. Hub (r <= r_hub): mid-dark (~110).
- **Scale reference**: a solid white filled circle of known physical diameter `ref_diameter_mm` placed in the
  top-left corner, fully outside the impeller. Analyzer may alternatively be told `outer_diameter_mm` directly.
- Image y axis points down (OpenCV). Because of this, a blade sweeping toward +theta in the math frame appears
  *clockwise* on screen. synth must draw using `polar_to_xy` with y negated (`py = cy - y*s`) so that the
  math frame is y-up; analyze must apply the same convention when unwrapping (use `cv2.warpPolar` and then
  flip/offset so that theta increases CCW in the y-up frame). The round-trip test will catch a sign error:
  a wrong sign yields beta ~ 180 - beta.

## Module: `fanscan/synth.py`
```
def render(params: ImpellerParams, px_per_mm: float = 4.0, ref_diameter_mm: float = 30.0,
           noise_sigma: float = 0.0, blur_sigma: float = 0.0, rotation_deg: float = 0.0,
           center_jitter_px: tuple[float, float] = (0, 0), seed: int = 0) -> tuple[np.ndarray, dict]
```
Returns (gray uint8 image, ground-truth dict) where the dict has `px_per_mm, center_xy, ref_center_xy,
ref_diameter_mm, rotation_deg` and `params` (params.to_dict()). Image size derives from D2 and px_per_mm with
margin for the reference circle. Draw blades via `offset_camber` polygons with `cv2.fillPoly` (anti-aliased,
LINE_AA) each rotated by `2*pi*i/Z + rotation`. Then optional Gaussian blur and additive Gaussian noise.
Also `def save(path, img)` and `__main__` entry that writes a sample PNG + JSON.

## Module: `fanscan/analyze.py`
```
@dataclass
class AnalysisResult:
    params: ImpellerParams
    px_per_mm: float
    center_xy: tuple[float, float]
    confidence: dict[str, float]      # 0..1 per parameter
    debug: dict[str, Any]             # intermediate arrays/values for debugging/overlay

def analyze(img: np.ndarray, *, ref_diameter_mm: float | None = None,
            outer_diameter_mm: float | None = None,
            blade_width_mm: float | None = None,
            blade_count_hint: int | None = None) -> AnalysisResult
def draw_overlay(img, result) -> np.ndarray   # BGR image with detected circles / blade traces drawn
```
Algorithm (all steps must be robust to the noise/blur levels in SPEC tests):
1. **Scale**: if `ref_diameter_mm` given, find the white reference circle (threshold > 240, contours, pick the
   most circular blob with area in a plausible range that is NOT the impeller) -> px_per_mm. Else require
   `outer_diameter_mm` and compute scale from the detected outer radius. If neither: raise ValueError.
2. **Center + outer radius**: threshold backplate vs background (Otsu on the image with the reference circle
   masked out), largest contour, `cv2.minEnclosingCircle` and/or least-squares circle fit of the contour.
   Refine center with the radial-symmetry of the blade pattern if cheap; minimal is fine.
3. **Radial profile**: `cv2.warpPolar` to (r, theta) with r in px, theta 0..2pi. Compute mean intensity vs r
   and the angular *variance* vs r. The blade annulus is where angular variance is high: r1 = inner edge of
   the high-variance band, r2 = outer edge (should match step 2). Hub radius: inner plateau where mean
   intensity ~110 (dark) and variance ~0 — detect by the step in the mean profile below r1.
4. **Blade count Z**: angular intensity profile at several radii in [r1+0.15*(r2-r1), r2-0.15*(r2-r1)];
   FFT each, take the dominant non-zero harmonic; majority vote across radii. Use `blade_count_hint` only to
   break ties. Z range 2..64.
5. **Blade tracing**: in the polar image, for each radius row in the annulus find the Z angular minima
   (blade centres) — e.g. smooth the row, find local minima with `scipy.signal.find_peaks` on the negated row,
   keep Z strongest. Link blades across rows by nearest-angle continuity (unwrap across 2pi). Result: Z traces
   of (r_px, theta_rad). Convert r to mm. Fit each with `geometry.fit_linear_beta`; report the median beta1,
   beta2 across blades and store per-blade values in debug. Sign: handle per the Image convention section.
6. **Blade thickness**: width of the dark ridge at half-depth, at mid radius, averaged over blades, corrected
   for the fact that the blade crosses the circle obliquely: `t = w_angular_px * r_mid * sin(beta_mid)`.
7. **Blade width (axial)**: NOT observable in an axial photo. Use `blade_width_mm` if given, else estimate
   `0.2 * D2` and set confidence 0.1 with a note in `params.meta["warnings"]`.
8. Fill `params.meta` with `source: "fanscan.analyze"`, `px_per_mm`, `warnings`, and per-blade betas.
Expose `__main__` that analyzes a PNG and prints JSON + writes an overlay PNG.

## Module: `fanscan/cad.py`
```
def build_impeller(params: ImpellerParams) -> cadquery.Workplane
def export_step(params: ImpellerParams, path: str | Path) -> Path
def export_stl(params: ImpellerParams, path) -> Path   # for quick viewing
```
Geometry (z up, backplate on the bottom):
- Backplate: cylinder r = r2 + 2 mm, height = backplate_thickness, z in [0, tb].
- Hub: cylinder r = r_hub, height = hub_height or (blade_width + tb), from z = 0.
- Blades: for i in range(Z): polygon = `offset_camber(r_mm, theta, thickness, theta0 = 2*pi*i/Z)`;
  build with `cq.Workplane("XY").polyline(points).close().extrude(blade_width)` translated to z = tb; union.
  Camber line must use `geometry.camber_line` with n >= 48 so the polygon is smooth.
- Optional shroud (params.shroud): ring r1..r2, thickness shroud_thickness, sitting at z = tb + blade_width.
- Union everything into one solid; `.val().isValid()` must be True. Export STEP with `cq.exporters.export`.
Performance target: 9-blade default builds in < 20 s.

## Module: `fanscan/cli.py`  (entry `python -m fanscan.cli ...`)
Subcommands:
- `synth  --out img.png [--params p.json] [--px-per-mm 4] [--noise 0] [--blur 0] [--rotation 0]`
   also writes `img.truth.json`.
- `analyze img.png --out params.json (--ref-diameter-mm X | --outer-diameter-mm X) [--blade-width-mm B]
   [--overlay overlay.png]`
- `build params.json --out impeller.step [--stl impeller.stl]`
- `pipeline [--params p.json] --out-dir DIR [--noise .. --blur .. --rotation ..]`
   synth -> analyze -> build in DIR, prints a comparison table (truth vs est, errors) and exits non-zero if
   blade_count mismatches or any diameter error > 3 % or any beta error > 4 deg.

## Tests (`tests/`, pytest)
- `test_geometry.py`: camber_line round trip via fit_linear_beta (|err| < 0.1 deg); offset_camber returns a
  closed, non-self-intersecting polygon for default params.
- `test_synth.py`: render default -> image dtype/size sane; the reference circle exists; rotating by
  360/Z deg gives an (almost) identical image.
- `test_analyze.py` (parametrised over Z in {6, 9, 12, 16}, beta2 in {30, 45, 60, 90}, noise in {0, 8},
  blur in {0, 1.5}, rotation in {0, 13}): synth -> analyze; assert Z exact, D2/D1/hub within 3 %, beta1/beta2
  within 4 deg, thickness within 30 %. Keep the full grid under ~60 s total; use `pytest.mark.parametrize`.
- `test_cad.py`: build default -> solid valid; STEP file written and starts with `ISO-10303-21`; volume is
  within 10 % of an analytic estimate (backplate + hub + Z * blade_len * t * b).
