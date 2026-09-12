"""Rendering: heat map and evidence frames.

The heat map is coloured by the density bands themselves rather than by a
relative scale. A relative ramp always shows a hot spot -- even in an empty
square -- which trains an operator to ignore it. Keying the colour to
persons/m2 means the picture and the panel agree, and an empty scene looks
empty.

Density is smoothed over roughly a metre of ground before colouring, using the
calibration, so the perspective does not make the far end of the frame look
sparse when it is not.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Anchors at the band edges, so a colour change is a band change.
# (density p/m2, B, G, R)
BAND_COLOURS = (
    (0.0, (150, 200, 120)),  # calm green
    (2.0, (120, 220, 235)),  # busy, pale yellow
    (3.0, (60, 170, 250)),  # crowded, orange
    (4.0, (55, 75, 240)),  # very crowded, red
    (5.0, (90, 40, 170)),  # crush risk, dark crimson
    (7.0, (120, 30, 120)),  # off the scale
)
LUT_MAX_DENSITY = 7.0
LUT_SIZE = 256


def _build_lut() -> np.ndarray:
    """256-entry BGRA lookup from density to colour and opacity."""
    lut = np.zeros((LUT_SIZE, 4), dtype=np.uint8)
    densities = np.linspace(0.0, LUT_MAX_DENSITY, LUT_SIZE)
    stops = np.array([d for d, _ in BAND_COLOURS])
    colours = np.array([c for _, c in BAND_COLOURS], dtype=np.float64)

    for ch in range(3):
        lut[:, ch] = np.interp(densities, stops, colours[:, ch]).astype(np.uint8)

    # Opacity: invisible when empty, readable by 'busy', never fully opaque --
    # the operator must always be able to see the scene through it.
    alpha = np.clip(densities / 1.8, 0, 1) * 165 + np.clip((densities - 1.8) / 2.5, 0, 1) * 70
    lut[:, 3] = np.clip(alpha, 0, 235).astype(np.uint8)
    lut[0, 3] = 0
    return lut


_LUT = _build_lut()


def density_per_m2_map(
    density_count: np.ndarray,
    m2_per_px: np.ndarray,
    smooth_m: float = 0.75,
    downscale: int = 1,
) -> np.ndarray:
    """Convert a per-pixel count map into a smoothed persons/m2 field.

    ``downscale`` reduces resolution before smoothing. The field is blurred
    over about a metre of ground either way, so the detail lost is detail the
    output never had -- but the Gaussian gets four times cheaper per step, and
    this runs on every published frame.
    """
    counts = density_count.astype(np.float32)
    areas = m2_per_px
    if downscale > 1:
        h, w = counts.shape
        size = (max(w // downscale, 1), max(h // downscale, 1))
        total = float(counts.sum())
        counts = cv2.resize(counts, size, interpolation=cv2.INTER_AREA)
        # INTER_AREA averages; a count map has to be rescaled back to its total.
        if counts.sum() > 1e-9:
            counts *= total / float(counts.sum())
        # Each coarse pixel covers downscale^2 fine pixels, so it covers that
        # much more ground. Averaging then multiplying keeps the two maps in
        # the same units -- getting this wrong scales the whole heat map.
        areas = cv2.resize(areas.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        areas = areas * float(downscale * downscale)

    finite = areas[np.isfinite(areas) & (areas > 0)]
    px_per_m = 1.0 / np.sqrt(np.median(finite)) if finite.size else 32.0 / max(downscale, 1)
    sigma = float(np.clip(smooth_m * px_per_m, 3.0, 60.0))
    k = int(sigma * 3) | 1  # odd kernel

    smoothed = cv2.GaussianBlur(counts, (k, k), sigma)
    safe_area = np.where(areas > 1e-9, areas, 1e-9)
    return (smoothed / safe_area).astype(np.float32)


def render_heat_png(
    density_m2: np.ndarray,
    downscale: int = 2,
) -> tuple[bytes, float]:
    """Encode a density field as a translucent BGRA PNG for the canvas overlay.

    Returned at reduced resolution: the field has already been smoothed over
    about a metre, so full resolution carries no extra information and costs
    bandwidth on every frame.
    """
    field = density_m2
    if downscale > 1:
        h, w = field.shape
        field = cv2.resize(field, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)

    idx = np.clip(field / LUT_MAX_DENSITY * (LUT_SIZE - 1), 0, LUT_SIZE - 1).astype(np.uint8)
    bgra = _LUT[idx]
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        return b"", float(density_m2.max())
    return buf.tobytes(), float(density_m2.max())


def encode_jpeg(frame_bgr: np.ndarray, long_edge: Optional[int] = None, quality: int = 78) -> bytes:
    """JPEG for the operator display, optionally downscaled first.

    Evidence thumbnails go through here at a deliberately coarse size. The
    downscale is a privacy control, not a bandwidth one: it is applied before
    anything reaches disk.
    """
    img = frame_bgr
    if long_edge:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > long_edge:
            scale = long_edge / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def band_legend() -> list[dict]:
    """The colour key, generated from the same anchors the heat map uses."""
    labels = ["Free flow", "Busy", "Crowded", "Very crowded", "Crush risk"]
    out = []
    for (lo, colour), label in zip(BAND_COLOURS[:-1], labels):
        out.append({
            "from": lo,
            "label": label,
            "css": f"rgb({colour[2]},{colour[1]},{colour[0]})",
        })
    return out
