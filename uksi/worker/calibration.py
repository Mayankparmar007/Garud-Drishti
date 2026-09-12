"""Ground calibration: image pixels to real metres.

This is the step that turns a pixel count into a number a commander can act
on. Four clicked points on the ground, four known real dimensions, and every
zone gets an honest ``area_m2``.

Deliberately numpy-only -- no OpenCV -- so the geometry can be unit-tested
without the vision stack loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------


def homography_from_points(image_pts: Sequence[Sequence[float]],
                           world_pts: Sequence[Sequence[float]]) -> Array:
    """Direct linear transform for 4+ correspondences.

    Returns the 3x3 matrix H mapping image pixels to world metres, normalised
    so H[2, 2] == 1.
    """
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("need at least 4 matching 2-D point pairs")

    rows = []
    for (x, y), (X, Y) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    A = np.asarray(rows, dtype=np.float64)

    _, _, vt = np.linalg.svd(A)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate calibration points")
    return h / h[2, 2]


def project(H: Array, pts: Sequence[Sequence[float]]) -> Array:
    """Apply a homography to an (N, 2) set of points."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((p.shape[0], 1))
    hom = np.hstack([p, ones]) @ H.T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return hom[:, :2] / w


def shoelace(poly: Array) -> float:
    """Absolute polygon area for an (N, 2) vertex list."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_area_m2(H: Array, polygon_px: Sequence[Sequence[float]]) -> float:
    """Real ground area of a polygon drawn in image pixels."""
    return shoelace(project(H, polygon_px))


def pixel_area_map(H: Array, width: int, height: int) -> Array:
    """Square metres of ground covered by each image pixel.

    For a projective map the local area scale is |det(H)| / |w|^3 where w is
    the homogeneous denominator at that pixel. Used to render the heat map in
    real units rather than arbitrary intensity -- pixels near the horizon
    cover far more ground than pixels underfoot.
    """
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    w = H[2, 0] * gx + H[2, 1] * gy + H[2, 2]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return np.abs(np.linalg.det(H)) / np.abs(w) ** 3


def displacement_m(H: Array, points_px: Array, flow_px: Array) -> Array:
    """Convert per-pixel image displacements into ground displacements.

    Projecting both endpoints and differencing is exact, which matters:
    a single scalar metres-per-pixel is wrong everywhere except the centre of
    a tilted view.
    """
    p0 = project(H, points_px)
    p1 = project(H, np.asarray(points_px, dtype=np.float64) + np.asarray(flow_px, dtype=np.float64))
    return p1 - p0


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def polygon_mask(polygon_px: Sequence[Sequence[float]], width: int, height: int) -> Array:
    """Boolean mask of pixels inside a polygon (even-odd ray casting).

    Computed once per scene-config change and cached, so the O(w*h*verts)
    cost never lands in the per-frame budget.
    """
    poly = np.asarray(polygon_px, dtype=np.float64).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=bool)
    if poly.shape[0] < 3:
        return mask

    x0 = max(int(np.floor(poly[:, 0].min())), 0)
    x1 = min(int(np.ceil(poly[:, 0].max())) + 1, width)
    y0 = max(int(np.floor(poly[:, 1].min())), 0)
    y1 = min(int(np.ceil(poly[:, 1].max())) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return mask

    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros(gx.shape, dtype=bool)

    n = poly.shape[0]
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if yi == yj:
            continue
        straddles = (yi > gy) != (yj > gy)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (gy - yi) / (yj - yi) + xi
        inside ^= straddles & (gx < x_cross)

    mask[y0:y1, x0:x1] = inside
    return mask


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    """Wraps a homography, or a flat fallback scale when uncalibrated.

    An uncalibrated scene still runs end to end -- it just reports against an
    assumed ground sample distance and the UI flags every figure as
    indicative. Never silently presents an assumed scale as a measured one.
    """

    H: Optional[Array] = None
    fallback_m_per_px: float = 0.05

    @classmethod
    def from_points(cls, image_pts, world_pts) -> "Calibrator":
        return cls(H=homography_from_points(image_pts, world_pts))

    @classmethod
    def uncalibrated(cls, m_per_px: float = 0.05) -> "Calibrator":
        return cls(H=None, fallback_m_per_px=m_per_px)

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def _effective_H(self) -> Array:
        if self.H is not None:
            return self.H
        s = self.fallback_m_per_px
        return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])

    def area_m2(self, polygon_px) -> float:
        return polygon_area_m2(self._effective_H(), polygon_px)

    def pixel_areas(self, width: int, height: int) -> Array:
        return pixel_area_map(self._effective_H(), width, height)

    def speeds_ms(self, points_px: Array, flow_px: Array, dt_s: float) -> Array:
        """Per-point ground speed in m/s from image-space flow vectors."""
        if dt_s <= 0:
            return np.zeros(len(points_px), dtype=np.float64)
        return displacement_m(self._effective_H(), points_px, flow_px) / dt_s
