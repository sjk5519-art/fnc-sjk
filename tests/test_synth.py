"""Tests for fanscan.synth (synthetic axial-view renderer)."""
import math

import numpy as np
import pytest

from fanscan import synth
from fanscan.params import DEFAULT_PARAMS

P = DEFAULT_PARAMS
PX_PER_MM = 4.0
REF_D = 30.0


@pytest.fixture(scope="module")
def rendered():
    return synth.render(P, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D)


def test_image_sane(rendered):
    img, truth = rendered
    assert img.dtype == np.uint8
    assert img.ndim == 2
    h, w = img.shape
    # must contain the backplate disc (r2 + rim) plus some margin
    min_side = 2 * (P.r2 + 2.0) * PX_PER_MM
    assert w >= min_side and h >= min_side
    assert w <= 1.6 * min_side and h <= 1.6 * min_side
    cx, cy = truth["center_xy"]
    assert 0 < cx < w and 0 < cy < h
    assert truth["px_per_mm"] == PX_PER_MM
    assert truth["ref_diameter_mm"] == REF_D
    assert truth["rotation_deg"] == 0.0
    assert truth["params"]["blade_count"] == P.blade_count
    for key in ("px_per_mm", "center_xy", "ref_center_xy", "ref_diameter_mm", "rotation_deg", "params"):
        assert key in truth

    # intensity levels of the convention are all present
    assert np.any(img < 50)                      # background
    assert np.any((img > 180) & (img < 230))     # backplate
    assert np.any((img > 55) & (img < 90))       # blades
    assert np.any((img > 95) & (img < 125))      # hub
    assert np.any(img > 240)                     # reference circle


def test_reference_circle_exists(rendered):
    import cv2

    img, truth = rendered
    mask = (img > 240).astype(np.uint8)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    assert n == 2, "exactly one white blob (the reference circle) expected"
    area = stats[1, cv2.CC_STAT_AREA]
    r_px = 0.5 * REF_D * PX_PER_MM
    assert area == pytest.approx(math.pi * r_px ** 2, rel=0.03)
    assert stats[1, cv2.CC_STAT_WIDTH] == pytest.approx(2 * r_px, abs=3)
    assert stats[1, cv2.CC_STAT_HEIGHT] == pytest.approx(2 * r_px, abs=3)
    cxr, cyr = centroids[1]
    assert (cxr, cyr) == pytest.approx(truth["ref_center_xy"], abs=1.0)
    # top-left corner and fully outside the backplate disc
    cx, cy = truth["center_xy"]
    assert cxr < cx and cyr < cy
    assert math.hypot(cxr - cx, cyr - cy) > (P.r2 + 2.0) * PX_PER_MM + r_px


def test_rotation_by_pitch_is_identity(rendered):
    img0, _ = rendered
    img1, truth1 = synth.render(P, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D,
                                rotation_deg=360.0 / P.blade_count)
    assert truth1["rotation_deg"] == pytest.approx(360.0 / P.blade_count)
    assert img0.shape == img1.shape
    diff = np.abs(img0.astype(np.int16) - img1.astype(np.int16))
    assert diff.mean() < 0.5
    assert (diff > 40).mean() < 0.002  # only anti-aliasing differences along blade edges


def test_rotation_by_half_pitch_differs(rendered):
    img0, _ = rendered
    img1, _ = synth.render(P, px_per_mm=PX_PER_MM, ref_diameter_mm=REF_D,
                           rotation_deg=180.0 / P.blade_count)
    diff = np.abs(img0.astype(np.int16) - img1.astype(np.int16))
    assert (diff > 40).mean() > 0.01


def test_noise_and_blur_apply():
    clean, _ = synth.render(P, px_per_mm=PX_PER_MM)
    noisy, _ = synth.render(P, px_per_mm=PX_PER_MM, noise_sigma=8.0, seed=1)
    assert noisy.dtype == np.uint8 and noisy.shape == clean.shape
    bg = clean < 50
    assert 4.0 < noisy[bg].std() < 12.0
    blurred, _ = synth.render(P, px_per_mm=PX_PER_MM, blur_sigma=1.5)
    assert blurred.shape == clean.shape
    # blur smooths edges: fewer "pure" intermediate-free pixels than the clean image
    assert np.abs(np.diff(blurred.astype(np.int16), axis=1)).max() < \
        np.abs(np.diff(clean.astype(np.int16), axis=1)).max()


def test_save_roundtrip(tmp_path, rendered):
    import cv2

    img, _ = rendered
    out = synth.save(tmp_path / "sub" / "img.png", img)
    back = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert back is not None and np.array_equal(back, img)
