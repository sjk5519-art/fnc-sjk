# Rim-gap "edge-on transition" (SPEC3) - independent principle validation

Independent re-derivation of the SPEC3 geometry plus a hand measurement of the transition on
`render3d` 'side' renders.  No production code; scratch scripts are listed at the end.
Realistic variant: `SHROUDED_PARAMS_FACTORY()` with `inner_diameter=120`, `hub_height=20`, **Z = 8**
(the real sample has 8 blades), beta1 = 30, t = 3 mm, b2 = 55 mm, r2 = 150 mm, backplate radius
r2 + 2 mm, shroud outer radius = r2.  Renders 1400x700, blur 0.8 px, default lighting, no clutter.

## 1. Summary

* The SPEC3 principle is right: the strip of blade face **adjacent to a trailing edge (TE)** switches from
  one side of the TE line to the other exactly where the TE's face plane contains the ray to the camera.
  For a far camera that is at `phi0 = beta2 - 180 deg` (CCW blades) and `|x0| / a = cos(beta2)`.
  Handedness: transition on the left of the rim centre (shroud up, camera outside) = CCW seen from the inlet.
* **Perspective is not a "slight" shift.**  Exactly (any elevation, camera on the -y side at horizontal distance
  `D_h` from the axis): `sin(beta2 - phi0) = r2 cos(beta2) / D_h`, i.e. phi0 moves toward the near point by
  `asin(r2 cos beta2 / D_h)`: 4.0 deg at d = 6 D2 and 12-15 deg at d = 1.5 D2 (beta2 = 25).  In the image
  *ratio* `|x0|/a` the azimuth shift and the magnification of the near TE largely cancel at d = 6 D2, eps = 5
  (residual < 0.3 deg with the right `a`), but not with a close or high camera: **+0.8 ... +1.0 deg at
  d = 1.5 D2, eps = 5 and +2.2 ... +3.8 deg at d = 1.5 D2, eps = 30** even with the right `a` (Table B).
  Converting the transition *azimuth* with `beta2 = phi0 + 180` is off by +2.3 ... +15 deg in every case.
  Measured on the grid, the formula exactly as specified (backplate rim as `a`, symmetric sign change) is
  **+2 ... +4 deg off for beta2 <= 35 everywhere, +1 ... +3 deg for 45..65 with a close camera** (Table A).
  Use the ray formula with D_h from focal_px/EXIF (section 2a) or synthesis.
* **Blade curvature does not move the transition** (the criterion is the TE's own face plane), but it makes the
  observable vanish quadratically: the adjacent strip just before the transition has width
  `c^2 / (4 q)` (c = angle between the face and the ray, q = half the curvature of the camber near the tip) and
  is sub-pixel for |c| < 2-3 deg at 1.8 px/mm.  A detector that locates the sign change *symmetrically* lands
  in the middle of that blind zone, +0.7 ... +1.5 deg late for beta2 <= 45 (measured; ~ -1 deg for 65 where
  the curvature sign flips, ~0 at 55) - the unbiased point is the onset of the strip on the branch that stays
  visible to depth (the +n face for dA/ds > 0), not the midpoint; and even that onset is +0.5 ... +1.5 deg late
  at 1.5-1.8 px/mm because the strip must grow to a pixel before it is seen.
* **The single-image sign-change estimate as written in SPEC3 is not viable at Z = 8**: linear interpolation of
  the signed width between the two bracketing TEs (45 deg apart) gives 10-20 deg errors (table, last column),
  because w(c) is strongly non-linear (quadratic self-occluded sliver on one side, visible-depth-limited strip
  on the other) and because the visible depth differs between the two TEs.  One image gives the handedness and a
  one-pitch bracket; beta2 to a few degrees needs the synthesis stage (or a width model that includes
  self-occlusion, perspective and occlusion depth - which is the synthesis stage).
* **A width detector needs a depth cue.**  With beta2 <= 45 the blade's *own concave face* (deeper, past the end
  face) is visible on the *other* side of every TE line, and so is the neighbour's face; a pure threshold /
  gradient reading of "strip width left / right of the TE" gets the wrong sign (hand reading of b25_e5_far:
  beta2 = 76 deg).  The adjacent strip is the one that spans the full gap height right at the TE
  (`analyze_rim_image`'s band-height proxy) - but that cue is absent at elevation 5 deg.
* Thickness: at the transition the blade shows its end face as a bright band `t` wide centred on the camber
  tip; the two strips attach to its two edges.  Use the band centre as x_TE: picking one edge biases x0 by t/2
  (1.3 deg at beta2 = 25, 0.6 deg at 65 for t = 3 mm, r2 = 150).  Using the *backplate* rim for `a` (r2 + 2 mm)
  biases beta2 by +1.6 (25) ... +0.4 deg (65).  The right scale is the projected r2 circle at the TE row
  height (a_row, between the backplate-edge and shroud-underside ellipses); the shroud rim alone is *worse*
  with a close elevated camera (it is nearer the camera: a_shroud is 4 % above a_row at d = 1.5 D2, eps = 30).
* Shroud overhang / backplate: with the camera above the gap at elevation eps, the ray through a gap row at
  height h above the backplate shows the backplate top beyond a depth `h / tan(eps)` (47.6 mm at the mid-row,
  eps = 30), and the rows near the shroud lose depth to the rim at grazing azimuths.  This clips the strips to
  wedges but does not move the transition; it does change every strip width, which again breaks width
  interpolation.  The gap mid-line is an **elliptical arc**, not an image row - scan along the arc.

## 2. Geometry, re-derived

World frame: z up, origin on the axis at the backplate bottom, impeller frame.  Math azimuth phi (+x axis = 0,
CCW).  Camera on the -y side (`azimuth_deg = 270` in `render3d.camera_preset('side')`), elevation eps,
`up = z`, so image x = world +x (right = f x up).  Blades sweep +theta as r grows (CCW), beta measured from the
tangential direction.

At a rim point P(phi) = (r2 cos phi, r2 sin phi, z):
```
t = (-sin phi, cos phi)          radial r = (cos phi, sin phi)
e = cos(beta2) t + sin(beta2) r = (sin A, cos A),      A = beta2 - phi      (blade exit direction)
n = (-cos A, sin A)                                    (face normal, points to the +theta side: n.t = sin beta2 > 0)
```
Camera centre C = (0, -D_h, z_c), `D_h = r2 + d cos(eps)` for the preset ('distance' d is measured from the
near rim point), `z_c = tb + b2/2 + d sin(eps)`.

**(a) Edge-on condition, exact perspective.**  The face at P is edge-on when the ray C->P lies in the face plane:
`(P - C) . n = 0`.  n has no z component, so the elevation and the row height drop out:
```
(P - C).n = -r2 cos(phi) cos(A) + (r2 sin(phi) + D_h) sin(A) = -r2 cos(beta2) + D_h sin(beta2 - phi)
=>  sin(beta2 - phi0) = r2 cos(beta2) / D_h,       phi0 = beta2 - 180 + asin(r2 cos(beta2) / D_h)   (near side)
```
D_h -> inf gives SPEC3's `phi0 = beta2 - 180`.  The shift `asin(r2 cos beta2 / D_h)` is 4.0 deg (d = 6 D2,
beta2 = 25), 3.1 deg (45), 1.9 deg (65) and 12.5-14.6 / 10.2 / 6.1 deg at d = 1.5 D2.  Verified against a
numeric root of the same equation with the true `Camera` (identical to 1e-3 deg).

Image position: `x0 = f X_c / Z_c` of P(phi0); the rim semi-major axis `a` is the extent of the projected rim
at phi = 0 / 180 where `Z_c` is larger than at phi0 (the near-left TE is closer to the camera by about
`r2 (1 - sin(beta2 + shift)) cos(eps)`).  So `|x0| / a = cos(beta2 + shift) * Z_c(phi=0) / Z_c(phi0) * r2 / r_a`
(r_a = radius of the circle used for a: r2 + 2 for the backplate rim).  The magnification nearly cancels the
azimuth shift at d = 6 D2, eps = 5 (with r_a = r2 at the TE row the exact ray transition gives
`cos^-1(|x0|/a)` = 25.1 / 35.1 / 45.1 / 55.1 / 65.1 for beta2 = 25 ... 65) but not with a close or high
camera (25.8 ... 65.7 at d = 1.5 D2, eps = 5; 26.1 ... 65.6 at d = 6 D2, eps = 30; 28.8 / 38.7 / 48.4 / 57.9 /
67.2 at d = 1.5 D2, eps = 30; Table B).
With a known camera (`D_h` from `focal_px`: `D_h ~ f_px * r2 / a + r2`, or EXIF) the correct recipe is:
measure x0, find phi0 from the projection of the rim circle at the TE height (invert `x(phi)`, e.g. the
`x_of_phi` model in `shrouded.analyze_rim_image`), then `beta2` from `sin(beta2 - phi0) = r2 cos(beta2)/D_h`
(one-dimensional root; the far camera is the special case `beta2 = phi0 + 180`).

**(b) Blade curvature.**  Along the blade *inward* from the tip (arc length s),
```
dA/ds = cos(beta2)/r2 - sin(beta2) * (beta2 - beta1)/(r2 - r1)       [rad/mm, angles in rad]
      = +0.370 (beta2=25), +0.281 (35), +0.152 (45), -0.008 (55), -0.191 (65) deg/mm   (beta1 = 30, r1 = 60, r2 = 150)
```
The first term is the wrap of a log-spiral-like blade, the second the linear beta taper; they cancel near
beta2 ~ 52 deg for these parameters.  In the frame where the ray to the tip is +y (depth), a face point at depth
s sits at `x(s) = c s + q s^2`, `q = (dA/ds)/2`, c = angle of the tip face to the ray (c = 0 at the transition
of (a)).  Consequences:
* The *criterion* "which face is adjacent to the end face" flips exactly at c = 0: the +n face is front-facing
  near the tip iff c > 0 (for q > 0 it is also the concave face and stays visible to large depth); the -n face
  is front-facing near the tip iff c < 0.  Curvature does **not** shift the transition - it is a property of
  the tip tangent, i.e. of beta2 itself, not of a chord.
* The *observable* does not flip sharply.  For c < 0 and q > 0 the -n (convex) face self-occludes beyond
  s* = |c|/(2q); its projected width is `c^2/(4q)` - quadratic in c, 0.15 mm = 0.3 px at c = 2.5 deg for
  beta2 = 25 (q = 0.0032/mm).  The +n face for c > 0 grows linearly, `w ~ c L + q L^2`, with L the visible depth
  (set by the next blade, the hub, the shroud overhang or the backplate - section c).  A symmetric
  "zero crossing" detector therefore sits ~ half the blind zone on the c < 0 side; measured blind zones are
  1-3 deg (table column "ambiguous zone"), i.e. a bias of about +1 ... +2.5 deg in beta2 (c < 0 is phi_rel >
  phi0, i.e. |x0| too small).  Which local angle is measured: the tip tangent (beta2) when the onset of the
  far-side (+n) strip is used; roughly the chord to the visible depth, `beta2 + 0.4 L dA/ds`, when a
  width-symmetric criterion is used with a deep strip (L = 40 mm at beta2 = 25: +6 deg).
* The same curvature puts the blade's own concave face on the *far* side of the TE line at almost every
  azimuth (seen past the end face, with a depth jump of 20-60 mm): in `b25_e5_far` the TE at phi = -125 has a
  29 px convex strip on its left (adjacent) and a 35 px concave strip on its right (not adjacent).

**(c) Thickness and occlusion.**
* The TE is a flat end face of width t perpendicular to e; at the transition it is seen face-on as a bright band
  t px/mm wide, centred on the camber tip (checked: band centre within 0.5 px of the projected tip in all cases).
  The -n strip attaches to its left edge, the +n strip to its right edge.  Taking one edge as "the TE line"
  shifts x0 by t/2 = 1.5 mm = 1 % of r2: `d beta2 = (t/2) / (r2 sin beta2)` = 1.3 deg (25) ... 0.6 deg (65).
  Use the band centre, or the edge on the side where the strip is.
* Rim radius used for `a`: backplate rim = r2 + 2 mm -> `|x0|/a` low by 1.3 % -> beta2 high by
  `0.013 / tan(beta2)` rad = +1.6 (25), +0.75 (45), +0.35 (65) deg.  The shroud rim is at r2 here (and the TE
  sits under the shroud rim in the real sample too); a_shroud is the right scale for the TE row.  Note that with
  a close elevated camera the two rims have visibly different `a` (shroud 240.8 vs backplate 231.6 px at
  d = 1.5 D2, eps = 30) because the shroud rim is nearer the camera.
* Shroud overhang / backplate occlusion: camera above the gap at elevation eps.  The ray through a gap row at
  height h above the backplate reaches the backplate top at horizontal depth `h / tan(eps)` (mid-row, h = 27.5:
  314 / 103 / 47.6 mm for eps = 5 / 15 / 30) - beyond that the lit backplate is seen *instead of* the deeper
  face (the wedge-shaped strips in the eps = 30 renders).  Rows near the shroud lose depth to the shroud rim
  where the ray grazes the rim circle (a point delta_r inside the rim is reached only after a chord
  ~ sqrt(2 r2 delta_r) during which the ray must stay below the shroud underside).  Measured visible depth of
  the adjacent strip (s_max, mm, `b25`, far camera, strips of TE k2/k3/k4):
  eps = 5: 67/54/119 at every row; eps = 15: top row 16/53/119, mid 66/57/117, bottom 34/34/63;
  eps = 30: top 1/61/110, mid 30/47/85, bottom 16/15/39.  The TE and the first 10-15 mm of face next to it
  stay visible at the mid-row in every case of the grid, so the transition itself is unaffected; the widths
  are.
* The gap mid-line is an elliptical arc (b/a = sin eps); at eps = 30 a straight image row through the near
  mid-gap point leaves the gap 150 px from the centre.  Every TE must be read on the row of its own mid-gap point
  (or the strip rectified along the arc, as SPEC3's synthesis stage says).

## 3. Measurement on renders

Scratch directory: `/tmp/claude-0/-home-user-fnc-sjk/ae3fa800-23db-5c9b-affd-8179edd5d6bf/scratchpad/v03/validate/`
(`render_grid.py`, `measure.py`, `sweep.py`, `hand.py`, `crops.py`, `postproc.py`; renders + id/depth arrays in
`renders/`, meshes/renders also in the tests render cache).  Grid: beta2 {25, 35, 45, 55, 65} x elevation
{5, 15, 30} x distance {far = 6 D2 = 1800 mm, close = 1.5 D2 = 450 mm} (the preset's `distance`, measured from
the near rim point; D_h = 1943 / 1945 / 1709 mm and 598 / 585 / 540 mm).

Per image, on each TE's own mid-gap row (gap = projected shroud-rim underside to backplate edge):
1. **Hand/grey reading** (`hand.py`, no 3-D truth): |gx| of the 1.2 px blurred image, edges = peaks > 0.25 max,
   TE = the bright end-face band, strip width on each side = distance to the next edge.  Reproduces the exact
   adjacent width on the correct side (b25_e5_far, k3: 30 px vs 29 exact) but also finds a "strip" on the wrong
   side every time (the deep concave face / neighbour face), see section 1.
2. **Exact adjacent strips** (`measure.py`): pixels are back-projected through the truth camera, assigned to a
   blade by the camber line, and described in the tip frame (inward distance s, face sign).  The adjacent strip
   is the run of same-blade pixels continuing from the end face without a depth jump (|ds| < 12 mm/px);
   signed width `w = (xa + xb) - 2 x_TE`.
3. **Transition by azimuth sweep** (`sweep.py`): the camera azimuth is rotated (equivalent to turning the
   impeller) until the TE right of the transition flips from "-n face adjacent" (state -1) to "+n face
   adjacent" (state +1); bisection to 0.05 deg on both ends of the ambiguous zone (neither face resolvable).
   x0 = image x of that TE (rel. to the backplate ellipse centre), a = backplate ellipse semi-major axis from
   the truth.  Resolution of a: the fitEllipse of the projected circle; of x0: 1 px = 0.1-0.15 deg.
4. **Single-image interpolation** (what a sign-change detector with *exact* widths would do): linear
   interpolation of w between the two bracketing TEs of the azimuth-270 image.

## 4. Results

All angles in degrees.  d = camera distance from the near rim point (far = 6 D2 = 1800 mm, close =
1.5 D2 = 450 mm); D_h = horizontal camera-to-axis distance.  "measured x0" = image x of the transition found by
the azimuth sweep (unbiased boundary of the blind zone, section 2b), relative to the backplate ellipse centre.
"blind zone" = azimuth interval over which neither face next to the end face is resolvable (its midpoint is
what a symmetric sign-change detector returns).  "ray-formula prediction" = thin straight-tip transition of
section 2a projected with the true camera, divided by a_bp.  "single-image linear interp." = zero crossing of
the exact signed adjacent-strip width interpolated linearly between the two bracketing TEs of the azimuth-270
image (what SPEC3's sign-change step would give with perfect widths).

**Table A - the formula as specified in SPEC3 (`|x0| / a_rim = cos(beta2)`, a_rim = backplate ellipse).**

| beta2 | elev | d | a_bp px | cos(beta2) | measured x0/a_bp | beta2 from cos^-1 | error | blind zone (deg) | zone-midpoint beta2 | ray-formula prediction (a_bp) | single-image linear interp. beta2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 5 | far | 276 | 0.906 | 0.887 | 27.5 | **+2.5** | 2.5 | 28.8 | 26.5 | 44.2 |
| 25 | 5 | close | 230 | 0.906 | 0.874 | 29.1 | **+4.1** | 1.6 | 29.9 | 26.9 | 40.7 |
| 25 | 15 | far | 276 | 0.906 | 0.883 | 28.0 | **+3.0** | 1.6 | 28.8 | 26.5 | 44.1 |
| 25 | 15 | close | 230 | 0.906 | 0.881 | 28.3 | **+3.3** | 1.1 | 28.9 | 26.6 | 40.5 |
| 25 | 30 | far | 277 | 0.906 | 0.880 | 28.4 | **+3.4** | 1.6 | 29.2 | 26.9 | 36.8 |
| 25 | 30 | close | 232 | 0.906 | 0.878 | 28.6 | **+3.6** | 2.7 | 30.2 | 27.7 | 33.4 |
| 35 | 5 | far | 276 | 0.819 | 0.796 | 37.2 | **+2.2** | 1.3 | 37.9 | 36.1 | 37.2 |
| 35 | 5 | close | 230 | 0.819 | 0.792 | 37.6 | **+2.6** | 0.9 | 38.2 | 36.7 | 60.0 |
| 35 | 15 | far | 276 | 0.819 | 0.800 | 36.9 | **+1.9** | 2.2 | 38.0 | 36.1 | 58.0 |
| 35 | 15 | close | 230 | 0.819 | 0.790 | 37.9 | **+2.9** | 1.9 | 39.0 | 36.7 | 60.0 |
| 35 | 30 | far | 277 | 0.819 | 0.786 | 38.2 | **+3.2** | 1.9 | 39.1 | 36.5 | 45.6 |
| 35 | 30 | close | 232 | 0.819 | 0.779 | 38.9 | **+3.9** | 1.8 | 40.0 | 37.9 | 49.1 |
| 45 | 5 | far | 276 | 0.707 | 0.698 | 45.7 | **+0.7** | 2.7 | 47.1 | 45.8 | 60.7 |
| 45 | 5 | close | 230 | 0.707 | 0.688 | 46.5 | **+1.5** | 2.2 | 47.8 | 46.5 | 59.3 |
| 45 | 15 | far | 276 | 0.707 | 0.706 | 45.1 | **+0.1** | 2.7 | 46.5 | 45.8 | 60.6 |
| 45 | 15 | close | 230 | 0.707 | 0.685 | 46.7 | **+1.7** | 2.1 | 48.0 | 46.6 | 59.1 |
| 45 | 30 | far | 277 | 0.707 | 0.685 | 46.8 | **+1.8** | 1.6 | 47.6 | 46.3 | 55.6 |
| 45 | 30 | close | 232 | 0.707 | 0.675 | 47.5 | **+2.5** | 2.3 | 49.0 | 47.8 | 57.4 |
| 55 | 5 | far | 276 | 0.574 | 0.565 | 55.6 | **+0.6** | 2.9 | 55.6 | 55.6 | 59.9 |
| 55 | 5 | close | 230 | 0.574 | 0.552 | 56.5 | **+1.5** | 3.1 | 56.5 | 56.3 | 56.9 |
| 55 | 15 | far | 276 | 0.574 | 0.565 | 55.6 | **+0.6** | 2.9 | 55.6 | 55.6 | 59.9 |
| 55 | 15 | close | 230 | 0.574 | 0.551 | 56.5 | **+1.5** | 2.9 | 56.5 | 56.4 | 56.8 |
| 55 | 30 | far | 277 | 0.574 | 0.561 | 55.9 | **+0.9** | 2.6 | 55.9 | 56.0 | 58.0 |
| 55 | 30 | close | 232 | 0.574 | 0.533 | 57.8 | **+2.8** | 1.4 | 57.8 | 57.5 | 57.7 |
| 65 | 5 | far | 276 | 0.423 | 0.409 | 65.9 | **+0.9** | 1.9 | 64.9 | 65.4 | 68.9 |
| 65 | 5 | close | 230 | 0.423 | 0.402 | 66.3 | **+1.3** | 1.9 | 65.1 | 66.0 | 69.7 |
| 65 | 15 | far | 276 | 0.423 | 0.409 | 65.9 | **+0.9** | 1.9 | 64.9 | 65.4 | 65.4 |
| 65 | 15 | close | 230 | 0.423 | 0.403 | 66.2 | **+1.2** | 1.9 | 65.0 | 66.1 | 67.2 |
| 65 | 30 | far | 277 | 0.423 | 0.407 | 66.0 | **+1.0** | 1.9 | 65.0 | 65.7 | 63.1 |
| 65 | 30 | close | 232 | 0.423 | 0.396 | 66.7 | **+1.7** | 1.3 | 65.9 | 66.9 | 64.4 |

**Table B - corrections.**  a_row = semi-major axis of the projected r2 circle at the TE row height (between
the backplate and the shroud-underside ellipses).  "ray-formula inversion" = the recipe of section 2a: measured
x0 -> phi0 through the perspective rim projection, then `sin(beta2 - phi0) = r2 cos(beta2) / D_h`.
"far inversion" = `beta2 = phi0 + 180` from the same phi0.

| beta2 | elev | d | D_h mm | a_row px | measured x0/a_row -> beta2 | error | measured phi0 (deg) | ray-formula inversion beta2 | error | far inversion beta2 = phi0+180 | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 5 | far | 1943 | 272.3 | 26.2 | **+1.2** | -149.94 | 26.1 | **+1.1** | 30.1 | +5.1 |
| 25 | 5 | close | 598 | 227.9 | 28.0 | **+3.0** | -139.94 | 27.2 | **+2.2** | 40.1 | +15.1 |
| 25 | 15 | far | 1889 | 272.9 | 26.9 | **+1.9** | -149.36 | 26.6 | **+1.6** | 30.6 | +5.6 |
| 25 | 15 | close | 585 | 229.3 | 28.1 | **+3.1** | -140.10 | 26.6 | **+1.6** | 39.9 | +14.9 |
| 25 | 30 | far | 1709 | 274.8 | 27.7 | **+2.7** | -148.95 | 26.5 | **+1.5** | 31.1 | +6.1 |
| 25 | 30 | close | 540 | 234.1 | 29.7 | **+4.7** | -139.63 | 25.9 | **+0.9** | 40.4 | +15.4 |
| 35 | 5 | far | 1943 | 272.3 | 36.3 | **+1.3** | -140.22 | 36.2 | **+1.2** | 39.8 | +4.8 |
| 35 | 5 | close | 598 | 227.9 | 36.9 | **+1.9** | -132.35 | 35.9 | **+0.9** | 47.6 | +12.6 |
| 35 | 15 | far | 1889 | 272.9 | 36.1 | **+1.1** | -140.49 | 35.8 | **+0.8** | 39.5 | +4.5 |
| 35 | 15 | close | 585 | 229.3 | 37.7 | **+2.7** | -131.87 | 36.2 | **+1.2** | 48.1 | +13.1 |
| 35 | 30 | far | 1709 | 274.8 | 37.7 | **+2.7** | -139.29 | 36.7 | **+1.7** | 40.7 | +5.7 |
| 35 | 30 | close | 540 | 234.1 | 39.6 | **+4.6** | -131.08 | 35.9 | **+0.9** | 48.9 | +13.9 |
| 45 | 5 | far | 1943 | 272.3 | 45.0 | **+0.0** | -131.94 | 44.9 | **-0.1** | 48.1 | +3.1 |
| 45 | 5 | close | 598 | 227.9 | 46.0 | **+1.0** | -124.78 | 45.0 | **+0.0** | 55.2 | +10.2 |
| 45 | 15 | far | 1889 | 272.9 | 44.6 | **-0.4** | -132.47 | 44.3 | **-0.7** | 47.5 | +2.5 |
| 45 | 15 | close | 585 | 229.3 | 46.7 | **+1.7** | -124.43 | 45.1 | **+0.1** | 55.6 | +10.6 |
| 45 | 30 | far | 1709 | 274.8 | 46.4 | **+1.4** | -130.97 | 45.5 | **+0.5** | 49.0 | +4.0 |
| 45 | 30 | close | 540 | 234.1 | 48.1 | **+3.1** | -123.93 | 44.7 | **-0.3** | 56.1 | +11.1 |
| 55 | 5 | far | 1943 | 272.3 | 55.2 | **+0.2** | -122.41 | 55.1 | **+0.1** | 57.6 | +2.6 |
| 55 | 5 | close | 598 | 227.9 | 56.1 | **+1.1** | -116.58 | 55.2 | **+0.2** | 63.4 | +8.4 |
| 55 | 15 | far | 1889 | 272.9 | 55.2 | **+0.2** | -122.40 | 55.0 | **-0.0** | 57.6 | +2.6 |
| 55 | 15 | close | 585 | 229.3 | 56.5 | **+1.5** | -116.43 | 55.1 | **+0.1** | 63.6 | +8.6 |
| 55 | 30 | far | 1709 | 274.8 | 55.6 | **+0.6** | -122.23 | 54.9 | **-0.1** | 57.8 | +2.8 |
| 55 | 30 | close | 540 | 234.1 | 58.1 | **+3.1** | -115.60 | 55.3 | **+0.3** | 64.4 | +9.4 |
| 65 | 5 | far | 1943 | 272.3 | 65.6 | **+0.6** | -112.67 | 65.5 | **+0.5** | 67.3 | +2.3 |
| 65 | 5 | close | 598 | 227.9 | 66.0 | **+1.0** | -108.70 | 65.3 | **+0.3** | 71.3 | +6.3 |
| 65 | 15 | far | 1889 | 272.9 | 65.6 | **+0.6** | -112.67 | 65.4 | **+0.4** | 67.3 | +2.3 |
| 65 | 15 | close | 585 | 229.3 | 66.2 | **+1.2** | -108.69 | 65.1 | **+0.1** | 71.3 | +6.3 |
| 65 | 30 | far | 1709 | 274.8 | 65.8 | **+0.8** | -112.65 | 65.2 | **+0.2** | 67.3 | +2.3 |
| 65 | 30 | close | 540 | 234.1 | 67.0 | **+2.0** | -108.43 | 64.8 | **-0.2** | 71.6 | +6.6 |

Checks on the derivation: the measured transition agrees with the thin straight-tip ray prediction within
-0.7 ... +2.1 deg (25 of 30 cases within 1.2 deg; the larger ones are beta2 = 25/35 where the strip onset is
slowest) once both use the same `a` (Table A columns "beta2 from cos^-1" vs
"ray-formula prediction", and Table B "measured x0 / a_row" vs the prediction 25.1 / 35.1 / 45.1 / 55.1 / 65.1
at d = 6 D2, eps = 5).  The remaining +0.5 ... +1.5 deg is the pixel-resolution bias of the strip onset
(section 2b); it is positive because the strip becomes resolvable only once c is a fraction of a degree past
zero on the far-side-visible branch.  The blind-zone midpoint is a further +0.7 ... +1.5 deg late for
beta2 <= 45 (q > 0) and ~ -1 deg for 65 (q < 0), ~0 for 55 (q ~ 0), exactly as section 2b predicts.


## 5. Where the simple formula is off by more than 2 deg, and the correction

Error of the formula as specified (Table A, column "error"; positive = beta2 over-estimated):

| beta2 | far, eps 5 / 15 / 30 | close, eps 5 / 15 / 30 |
|---|---|---|
| 25 | +2.5 / +3.0 / +3.4 | +4.1 / +3.3 / +3.6 |
| 35 | +2.2 / +1.9 / +3.2 | +2.6 / +2.9 / +3.9 |
| 45 | +0.7 / +0.1 / +1.8 | +1.5 / +1.7 / +2.5 |
| 55 | +0.6 / +0.6 / +0.9 | +1.5 / +1.5 / +2.8 |
| 65 | +0.9 / +0.9 / +1.0 | +1.3 / +1.2 / +1.7 |

1. **beta2 <= 35 is off by 2-4 deg in every camera configuration**, including the far, low camera.  Budget at
   d = 6 D2, eps = 5, beta2 = 25: +1.6 deg from using the backplate rim (r2 + 2 mm) as `a`, +1.0 deg
   resolution bias of the strip onset, ~0 net perspective (azimuth shift and near-side magnification cancel).
   Correction: take `a` from the rim circle at the TE height (a_row: mean of the backplate-edge and
   shroud-underside ellipses, or the backplate ellipse scaled by r2 / (r2 + overhang) when the overhang is
   known; it is 2 mm in the model, unknown on the real part) -> Table B "measured x0 / a_row": +1.2 / +1.3 /
   0.0 / +0.2 / +0.6 at d = 6 D2, eps = 5.  Do not use the shroud rim alone with an elevated close camera: it
   is nearer the camera and its `a` is 4 % larger than a_row at eps = 30, d = 1.5 D2 (+4 deg).
2. **Close camera (d = 1.5 D2) and/or elevation 30 deg**: with a_row the error is still +1.0 ... +4.7 deg
   (beta2 <= 45 at eps = 30, close: +3.1 ... +4.7; far eps = 30: +1.4 ... +2.7 for beta2 <= 35).  This is real
   perspective: the ratio `|x0|/a` is not a function of beta2 alone once the camera is close or high.
   Correction: invert the image position with the perspective rim model (phi0) and solve
   `sin(beta2 - phi0) = r2 cos(beta2) / D_h` (Table B "ray-formula inversion": -0.7 ... +2.2 deg, mean +0.6,
   29 of 30 cases within 2 deg; the one outlier, b25 close eps 5, is the onset-resolution bias at the lowest
   px/mm).  D_h = r2 + d cos(eps) needs the camera distance: `focal_px * r2 / a_row + r2` from EXIF focal length
   and sensor width, or the two-rim parallax (shroud vs backplate ellipse) which is exactly the perspective
   cue `pose.estimate_perspective_scale` uses.
3. **Never convert the transition azimuth with `beta2 = phi0 + 180` (far-camera assumption)**: the azimuth
   shift `asin(r2 cos beta2 / D_h)` is +2.3 ... +6 deg at d = 6 D2 and +6 ... +15 deg at d = 1.5 D2 (Table B last
   column) - it is the *image ratio*, not the azimuth, in which perspective partly cancels.  If the implementer
   fits the TE sequence with `analyze_rim_image`'s `x_of_phi` model (xc, A, c) and reads phi0 from it, the
   ray formula is mandatory; with the far assumption one would have to widen the confidence interval to
   +-(r2 / D_h) rad, which for an unknown phone distance is +-10 deg.
4. **Symmetric sign-change localisation adds +0.7 ... +1.5 deg** for beta2 <= 45 (blind zone 1-3 deg wide;
   Table A "zone-midpoint").  Correction: locate the onset of the strip on the branch that is visible to large
   depth (the +n face for backward-curved tips with dA/ds > 0, the -n face for dA/ds < 0; at beta2 ~ 52 deg for
   these parameters both branches are linear and the midpoint is unbiased).  Since the sign of dA/ds depends on
   beta2 itself, do it iteratively or, simpler, accept the +-1.5 deg and widen the interval.
5. **Single-image linear interpolation of the signed width between neighbouring TEs is 10-20 deg off**
   (Table A last column: 44 for 25, 58-60 for 35 and 45, 57-60 for 55, 63-70 for 65) even with *exact* strip
   widths; the SPEC3 target "single image |err| <= 5 deg (sign-change)" is not reachable with Z = 8 by
   interpolation.  The width is not ~ sin(beta2 - phi_i) d_i: on the self-occluded branch it is quadratic
   (`c^2 / 4q`), on the other branch `c L + q L^2` with L set by occlusion, and L differs between the two
   bracketing TEs by a factor 2-5.  What one image does give robustly: the handedness and the pitch-wide
   bracket (which TE pair the transition lies between).  Three images at random azimuths narrow the bracket to
   ~ pitch/3 ~ 15 deg at best - still > 2 deg.  The "<= 2 deg with 3 images" target therefore needs the
   synthesis stage (or a width model that is the synthesis stage).
6. Thickness: +-1.3 deg (beta2 = 25) ... +-0.6 deg (65) if the TE line is one edge of the end-face band
   instead of its centre - keep the band centre.  Shroud overhang / backplate occlusion: no effect on the
   transition, large effect on widths (section 2c).


## 6. Notes for the implementer / reviewers

* **Adjacency needs a depth cue.**  At every azimuth there is a shaded strip on *both* sides of a TE line:
  the adjacent face on one side, and the blade's own concave face (seen past the end face, 20-60 mm deeper)
  or the neighbouring blade's face on the other.  A threshold / gradient width reading cannot tell them apart
  (section 3, item 1).  Cues that do work in the renders: the adjacent strip touches the end face without a
  brightness step and spans the full gap height at the TE; the deeper surface is clipped by the shroud
  overhang at the top (eps >= 10-15 deg) and starts with an intensity step.  At eps = 5 the overhang cue is
  gone (occlusion depth 314 mm) and only the brightness step / shading gradient remains.  The real sample
  (eps ~ 15) shows exactly the wedge-shaped strips of the eps = 15 / 30 renders.
* **Scan along the gap arc, per TE.**  The mid-gap line is an ellipse arc; at eps = 30 a straight row
  through the near mid-gap point is outside the gap 150 px from the centre, and even at eps = 15 the row
  near the backplate runs into the backplate top edge 60 px away.  Read every TE on the row of its own
  mid-gap point (project the TE azimuth onto the two rim arcs and take the mid-point), or rectify the strip
  along the arc as the synthesis stage does.
* **Which `a`.**  Measure both rim arcs (`pose.find_ellipses` returns them as two nearly parallel flat
  ellipses); use their mean semi-axis for the TE row and their difference (plus b2 / r2 from the gap height)
  as the perspective cue for D_h.  Backplate overhang beyond the blade tips is unknown on a real part (2 mm
  in the model = 1.6 deg at beta2 = 25); the shroud rim sits at the tips in the sample.
* **Handedness** is unambiguous: transition left of the rim centre (shroud up, camera outside) <=> blades
  sweep CCW seen from the inlet; a horizontally mirrored image moves it to the right.  Note the ray formula is
  symmetric, so the mirrored case gives the same beta2.
* **Realistic error budget for the simple (non-synthesis) estimate** on a phone photo (d ~ 1.5-3 D2,
  eps 10-20 deg, 1.5-2 px/mm): a_row and ray formula applied, TE band centre used -> +-2 deg systematic
  (onset bias, rim-radius uncertainty) plus the localisation error of the transition.  Without D_h (far
  assumption): add +3 ... +12 deg.  Without the adjacency cue: the sign itself is unreliable.
* **Synthesis stage (SPEC3)** handles everything above by construction (perspective, curvature, thickness,
  occlusion, the arc); it should compare rendered and photographed strips *per TE row along the arc* and the
  candidate grid must include the camera distance (or take it from the two-rim parallax), otherwise the
  perspective error of item 2 is baked into the match.
* Reproduction: `cd /home/user/fnc-sjk && . .venv/bin/activate && python <scratch>/render_grid.py && python
  <scratch>/sweep.py && python <scratch>/postproc.py`; `python <scratch>/hand.py b25_e5_far` prints the
  grey-level widths; `python <scratch>/crops.py b25_e5_far` writes an annotated gap crop (green = projected TE
  tips, red = ray-formula transition, blue = far formula).  Meshes for Z = 8, beta2 {25..65} are in the tests
  render cache (`tests/conftest.py` location) and take ~8-12 s each to build; renders 0.3-0.7 s.

