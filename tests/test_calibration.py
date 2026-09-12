"""Geometry tests. These run without OpenCV or torch on purpose."""

from __future__ import annotations

import numpy as np
import pytest

from uksi.worker.calibration import (
    Calibrator,
    homography_from_points,
    pixel_area_map,
    polygon_area_m2,
    polygon_mask,
    project,
    shoelace,
)


def test_shoelace_unit_square():
    assert shoelace([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_shoelace_is_orientation_independent():
    cw = [[0, 0], [0, 2], [2, 2], [2, 0]]
    ccw = list(reversed(cw))
    assert shoelace(cw) == pytest.approx(shoelace(ccw)) == pytest.approx(4.0)


def test_scale_only_homography():
    """100 px square known to be 10 m across -> 100 m2."""
    img = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world = [[0, 0], [10, 0], [10, 10], [0, 10]]
    H = homography_from_points(img, world)
    assert polygon_area_m2(H, img) == pytest.approx(100.0, rel=1e-6)


def test_subregion_area_scales_correctly():
    img = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world = [[0, 0], [10, 0], [10, 10], [0, 10]]
    H = homography_from_points(img, world)
    half = [[0, 0], [50, 0], [50, 100], [0, 100]]
    assert polygon_area_m2(H, half) == pytest.approx(50.0, rel=1e-6)


# A tilted drone view: the bottom of the frame is close to the camera and wide
# in pixels, the top is further away and compressed. Both edges are the same
# 20 m on the ground.
TILTED_IMG = [[20, 200], [380, 200], [300, 100], [100, 100]]
TILTED_WORLD = [[0, 0], [20, 0], [20, 15], [0, 15]]


def test_perspective_trapezoid_becomes_rectangle():
    H = homography_from_points(TILTED_IMG, TILTED_WORLD)
    assert polygon_area_m2(H, TILTED_IMG) == pytest.approx(300.0, rel=1e-6)

    corners = project(H, TILTED_IMG)
    np.testing.assert_allclose(corners, np.array(TILTED_WORLD, dtype=float), atol=1e-6)


def test_far_pixels_cover_more_ground_than_near_pixels():
    """The whole point of the per-pixel area map."""
    H = homography_from_points(TILTED_IMG, TILTED_WORLD)
    areas = pixel_area_map(H, 400, 220)
    near = areas[199, 200]  # bottom of the frame, close to the camera
    far = areas[101, 200]  # top of the frame, further away
    assert far > near * 1.5


def test_pixel_area_map_integrates_to_polygon_area():
    """Jacobian area map and the shoelace area must agree."""
    img, world = TILTED_IMG, TILTED_WORLD
    H = homography_from_points(img, world)
    mask = polygon_mask(img, 400, 220)
    integrated = float(pixel_area_map(H, 400, 220)[mask].sum())
    assert integrated == pytest.approx(polygon_area_m2(H, img), rel=0.02)


def test_polygon_mask_counts_interior_pixels():
    mask = polygon_mask([[10, 10], [30, 10], [30, 20], [10, 20]], 50, 40)
    assert mask.sum() == 200
    assert mask[15, 20] and not mask[5, 5]


def test_polygon_mask_handles_out_of_frame_polygon():
    mask = polygon_mask([[-50, -50], [-10, -50], [-10, -10], [-50, -10]], 40, 40)
    assert mask.sum() == 0


def test_polygon_mask_concave_shape():
    """An L. The notch must be excluded."""
    poly = [[0, 0], [40, 0], [40, 10], [10, 10], [10, 40], [0, 40]]
    mask = polygon_mask(poly, 50, 50)
    assert mask[5, 35]  # inside the horizontal arm
    assert mask[35, 5]  # inside the vertical arm
    assert not mask[35, 35]  # the notch


def test_speed_conversion_uses_real_ground_distance():
    img = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world = [[0, 0], [10, 0], [10, 10], [0, 10]]  # 0.1 m per px
    cal = Calibrator.from_points(img, world)
    pts = np.array([[50.0, 50.0]])
    flow = np.array([[3.0, 4.0]])  # 5 px over the interval
    v = cal.speeds_ms(pts, flow, dt_s=0.5)
    assert float(np.hypot(*v[0])) == pytest.approx(1.0)  # 0.5 m in 0.5 s


def test_uncalibrated_calibrator_still_produces_numbers():
    cal = Calibrator.uncalibrated(m_per_px=0.05)
    assert not cal.is_calibrated
    assert cal.area_m2([[0, 0], [100, 0], [100, 100], [0, 100]]) == pytest.approx(25.0)


def test_degenerate_points_are_rejected():
    with pytest.raises(ValueError):
        homography_from_points([[0, 0]] * 4, [[0, 0], [1, 0], [1, 1], [0, 1]])
