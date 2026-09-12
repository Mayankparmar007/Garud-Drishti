"""Movement: how fast, which way, how uniformly.

Farneback dense flow between sampled frames. Because the drone hovers, full
camera-motion compensation is unnecessary -- but wind jitter is real, so the
residual motion of the empty ground is measured and subtracted. If the whole
frame is crowd there is no static ground to measure against, and the
correction is skipped rather than guessed.

Two things are reported per zone and they are not the same number:

  mean speed  the average of the individual speeds -- how fast people move
  direction   the direction of the average velocity -- where the crowd goes

A zone where half the crowd walks north and half south has a real mean speed
and no meaningful direction. Collapsing those into one figure hides exactly
the situation worth seeing, so the disagreement is reported too, as variance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .calibration import Calibrator

# Sample the flow field on a grid rather than per pixel: the homography
# conversion is the expensive part and 8 px is far finer than a person.
GRID_STEP = 8

# Farneback is run at half resolution and the result scaled back up. Measured
# against simulator ground truth, per-zone speed is indistinguishable from the
# full-resolution answer (within 2% in free-flowing zones, identical inside a
# packed one) for a quarter of the cost. At drone altitude the flow field is
# far smoother than the pixel grid, so the detail being discarded is noise.
FLOW_SCALE = 0.5


@dataclass
class ZoneFlow:
    mean_speed_ms: float = 0.0
    flow_dir_deg: Optional[float] = None
    speed_variance: float = 0.0
    coherence: float = 0.0  # 1.0 = everyone agrees on a direction, 0 = churn


class FlowStage:
    """Holds the previous frame and a short history per zone."""

    def __init__(self, smooth_window: int = 5):
        self.prev_gray: Optional[np.ndarray] = None
        self.smooth_window = smooth_window
        self._history: dict[str, deque] = {}

    def reset(self) -> None:
        self.prev_gray = None
        self._history.clear()

    def flow_for(self, frame_bgr: np.ndarray, density: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Dense flow in full-resolution pixels since the previous call.

        Computed at FLOW_SCALE and resampled back, so the returned field is
        always in the frame's own pixel units and callers need not care.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if FLOW_SCALE != 1.0:
            gray = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE,
                              interpolation=cv2.INTER_AREA)

        prev, self.prev_gray = self.prev_gray, gray
        if prev is None or prev.shape != gray.shape:
            return None

        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        if FLOW_SCALE != 1.0:
            # Resampling moves the vectors; dividing by the scale restores
            # their magnitude in full-resolution pixels.
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / FLOW_SCALE
        return self._remove_jitter(flow, density)

    @staticmethod
    def _remove_jitter(flow: np.ndarray, density: Optional[np.ndarray]) -> np.ndarray:
        """Subtract the apparent motion of ground that should not be moving."""
        if density is None or density.shape != flow.shape[:2]:
            return flow
        static = density < (density.max() * 0.02 if density.max() > 0 else 1e-6)
        if static.sum() < static.size * 0.15:
            return flow  # not enough empty ground to trust the estimate
        jitter = np.median(flow[static], axis=0)
        if not np.all(np.isfinite(jitter)):
            return flow
        return flow - jitter.reshape(1, 1, 2)

    def zone_flow(
        self,
        zone_id: str,
        flow: Optional[np.ndarray],
        mask: np.ndarray,
        density: np.ndarray,
        cal: Calibrator,
        dt_s: float,
    ) -> ZoneFlow:
        """Aggregate flow inside one zone, weighted by where the people are."""
        result = self._measure(flow, mask, density, cal, dt_s)

        # Median over a short window. One bad Farneback frame -- a bird, a
        # compression artefact -- should not move an operator's display.
        hist = self._history.setdefault(zone_id, deque(maxlen=self.smooth_window))
        hist.append(result)
        speeds = [r.mean_speed_ms for r in hist]
        variances = [r.speed_variance for r in hist]
        coherences = [r.coherence for r in hist]
        dirs = [r.flow_dir_deg for r in hist if r.flow_dir_deg is not None]

        return ZoneFlow(
            mean_speed_ms=float(np.median(speeds)),
            flow_dir_deg=_circular_median(dirs) if dirs else None,
            speed_variance=float(np.median(variances)),
            coherence=float(np.median(coherences)),
        )

    @staticmethod
    def _measure(
        flow: Optional[np.ndarray],
        mask: np.ndarray,
        density: np.ndarray,
        cal: Calibrator,
        dt_s: float,
    ) -> ZoneFlow:
        if flow is None or dt_s <= 0 or not mask.any():
            return ZoneFlow()

        ys, xs = np.nonzero(mask[::GRID_STEP, ::GRID_STEP])
        if ys.size == 0:
            return ZoneFlow()
        ys, xs = ys * GRID_STEP, xs * GRID_STEP

        weights = density[ys, xs].astype(np.float64)
        total_w = weights.sum()
        if total_w <= 1e-9:
            # Nobody in the zone. Reporting the speed of empty pavement as
            # "the crowd has stopped" would be the wrong kind of wrong.
            return ZoneFlow()

        points = np.column_stack([xs, ys]).astype(np.float64)
        flow_px = flow[ys, xs].astype(np.float64)
        vel = cal.speeds_ms(points, flow_px, dt_s)  # (N, 2) m/s on the ground

        speeds = np.linalg.norm(vel, axis=1)
        mean_speed = float(np.average(speeds, weights=weights))

        mean_vec = np.average(vel, axis=0, weights=weights)
        spread = vel - mean_vec
        variance = float(np.average(np.einsum("ij,ij->i", spread, spread), weights=weights))

        # Direction is taken in image space: it is drawn on the operator's
        # screen, so screen orientation is the only frame that means anything.
        mean_flow_px = np.average(flow_px, axis=0, weights=weights)
        magnitude = float(np.linalg.norm(mean_flow_px))
        direction = None
        if magnitude > 0.25 and mean_speed > 0.05:
            direction = _bearing_from_image_vector(*mean_flow_px)

        coherence = float(np.linalg.norm(mean_vec) / mean_speed) if mean_speed > 1e-6 else 0.0
        return ZoneFlow(
            mean_speed_ms=mean_speed,
            flow_dir_deg=direction,
            speed_variance=variance,
            coherence=min(1.0, coherence),
        )


def _bearing_from_image_vector(vx: float, vy: float) -> float:
    """Degrees clockwise from straight up in the frame.

    Up-screen is 0, right is 90. Image y grows downward, hence the negation.
    """
    return float(np.degrees(np.arctan2(vx, -vy)) % 360.0)


def _circular_median(degrees: list[float]) -> float:
    """Median of headings. Averaging 350 and 10 must give 0, not 180."""
    rad = np.radians(degrees)
    mean = np.arctan2(np.median(np.sin(rad)), np.median(np.cos(rad)))
    return float(np.degrees(mean) % 360.0)
