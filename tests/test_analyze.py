"""Round-trip tests: synth -> analyze over a grid of impellers and image degradations.

The grid is Z x beta2 x noise x blur x rotation = 4*4*2*2*2 = 128 cases and
must stay under ~60 s in total, so the analyzer has roughly 0.4 s per image.
"""
import dataclasses
import itertools

import pytest

from fanscan import synth
from fanscan.params import DEFAULT_PARAMS

PX_PER_MM = 4.0
REF_D = 30.0

Z_VALUES = (6, 9, 12, 16)
BETA2_VALUES = (30.0, 45.0, 60.0, 90.0)
NOISE_VALUES = (0.0, 8.0)
BLUR_VALUES = (0.0, 1.5)
ROTATION_VALUES = (0.0, 13.0)

GRID = list(itertools.product(Z_VALUES, BETA2_VALUES, NOISE_VALUES, BLUR_VALUES, ROTATION_VALUES))
IDS = [f"Z{z}-b2_{b:g}-n{n:g}-blur{bl:g}-rot{r:g}" for z, b, n, bl, r in GRID]


def _analyze():
    # Lazy import: the analyzer is developed separately from synth/cad.
    from fanscan import analyze
    return analyze


@pytest.mark.parametrize("Z,beta2,noise,blur,rotation", GRID, ids=IDS)
def test_round_trip(Z, beta2, noise, blur, rotation):
    analyze = _analyze()
    truth = dataclasses.replace(DEFAULT_PARAMS, blade_count=Z, blade_angle_outlet=beta2, meta={})
    truth.validate()
    img, gt = synth.render(truth, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D,
                           noise_sigma=noise, blur_sigma=blur, rotation_deg=rotation,
                           seed=Z * 1000 + int(beta2))

    res = analyze.analyze(img, ref_diameter_mm=REF_D)
    est = res.params

    assert est.blade_count == Z
    for name in ("outer_diameter", "inner_diameter", "hub_diameter"):
        t, e = getattr(truth, name), getattr(est, name)
        assert abs(e - t) / t <= 0.03, f"{name}: truth {t} est {e}"
    assert abs(est.blade_angle_inlet - truth.blade_angle_inlet) <= 4.0, \
        f"beta1: truth {truth.blade_angle_inlet} est {est.blade_angle_inlet}"
    assert abs(est.blade_angle_outlet - truth.blade_angle_outlet) <= 4.0, \
        f"beta2: truth {truth.blade_angle_outlet} est {est.blade_angle_outlet}"
    t, e = truth.blade_thickness, est.blade_thickness
    assert abs(e - t) / t <= 0.30, f"thickness: truth {t} est {e}"
    # scale must agree with the renderer
    assert res.px_per_mm == pytest.approx(gt["px_per_mm"], rel=0.03)


def test_result_structure():
    analyze = _analyze()
    img, gt = synth.render(DEFAULT_PARAMS, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D)
    res = analyze.analyze(img, ref_diameter_mm=REF_D, blade_width_mm=DEFAULT_PARAMS.blade_width)
    assert res.params.blade_width == DEFAULT_PARAMS.blade_width
    assert isinstance(res.confidence, dict) and res.confidence
    assert all(0.0 <= v <= 1.0 for v in res.confidence.values())
    cx, cy = res.center_xy
    assert (cx, cy) == pytest.approx(gt["center_xy"], abs=3.0)
    assert res.params.meta.get("source") == "fanscan.analyze"
    overlay = analyze.draw_overlay(img, res)
    assert overlay.ndim == 3 and overlay.shape[:2] == img.shape


def test_scale_source_required():
    analyze = _analyze()
    img, _ = synth.render(DEFAULT_PARAMS, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D)
    with pytest.raises(ValueError):
        analyze.analyze(img)


def test_outer_diameter_scale():
    analyze = _analyze()
    img, _ = synth.render(DEFAULT_PARAMS, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D)
    res = analyze.analyze(img, outer_diameter_mm=DEFAULT_PARAMS.outer_diameter)
    assert res.params.blade_count == DEFAULT_PARAMS.blade_count
    assert res.px_per_mm == pytest.approx(PX_PER_MM, rel=0.03)
    assert res.params.inner_diameter == pytest.approx(DEFAULT_PARAMS.inner_diameter, rel=0.03)
