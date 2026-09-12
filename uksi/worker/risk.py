"""The risk engine.

A pure function over per-zone metric history. No I/O, no clock, no model --
call it with synthetic inputs and it is fully determined. That is the only way
to test logic whose real-world trigger condition you cannot obtain footage of.

The central claim: density alone does not predict a crush. A packed but
smoothly moving crowd is stable; the same density going stop-and-go is not.
The transition is what buys warning time, so ``flow_breakdown`` outranks raw
density when both fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from ..contract import Band, RiskLevel, Signal, Thresholds, Trend, band_for, worst

# Which signal gets to write the headline when several reach the same level.
# Earliest-warning first: a stalling crowd is more actionable than a number
# that is already high.
HEADLINE_PRIORITY = ("flow_breakdown", "density_trend", "convergence", "pressure", "density")

SIGNAL_LABELS = {
    "density": "Density",
    "density_trend": "Trend",
    "flow_breakdown": "Flow breakdown",
    "convergence": "Convergence",
    "pressure": "Pressure",
}

SUGGESTED_ACTION = {
    "density": "Hold inflow at the upstream barrier until density falls.",
    "density_trend": "Slow inflow now -- acting after the threshold is crossed is too late.",
    "flow_breakdown": "Movement is stalling. Stop inflow and open the nearest corridor.",
    "convergence": "Opposing streams meeting. Separate the flows before they lock.",
    "pressure": "Turbulent movement. Get eyes on this zone and prepare to hold inflow.",
}


@dataclass(frozen=True)
class Observation:
    """One sample of one zone. The only input the engine understands."""

    ts: float  # epoch seconds
    density: float
    count: float = 0.0
    mean_speed_ms: float = 0.0
    flow_dir_deg: Optional[float] = None
    speed_variance: float = 0.0


@dataclass
class ZoneAssessment:
    band: Band = "free"
    risk: RiskLevel = "green"
    trend: Trend = "steady"
    trend_rate_per_min: float = 0.0
    time_to_critical_s: Optional[float] = None
    pressure: float = 0.0
    convergence: float = 0.0
    headline: str = ""
    action: str = ""
    signals: list[Signal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _window(history: Sequence[Observation], window_s: float) -> list[Observation]:
    if not history:
        return []
    cutoff = history[-1].ts - window_s
    return [o for o in history if o.ts >= cutoff]


def _smooth(values: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving average. Wind jitter and model noise both live above this."""
    if k <= 1 or values.size < k:
        return values
    kernel = np.ones(k) / k
    return np.convolve(values, kernel, mode="valid")


def slope_per_min(samples: Sequence[Observation]) -> float:
    """Least-squares slope of density over time, in persons/m2 per minute.

    Returns 0 unless there is enough spread in time to mean anything -- a
    slope fitted across 2 seconds of a noisy signal is not a trend.
    """
    if len(samples) < 4:
        return 0.0
    t = np.array([o.ts for o in samples], dtype=np.float64)
    d = np.array([o.density for o in samples], dtype=np.float64)
    span = t[-1] - t[0]
    if span < 5.0:
        return 0.0

    d = _smooth(d, 3)
    t = t[len(t) - len(d):]  # keep the tail aligned after 'valid' convolution
    t = t - t[0]
    if np.allclose(t, t[0]):
        return 0.0

    slope_per_s = float(np.polyfit(t, d, 1)[0])
    return slope_per_s * 60.0


def speed_baseline(samples: Sequence[Observation]) -> float:
    """Recent unimpeded speed for this zone, robust to a couple of bad frames."""
    speeds = np.array([o.mean_speed_ms for o in samples], dtype=np.float64)
    if speeds.size == 0:
        return 0.0
    return float(np.percentile(speeds, 75))


def _angle_opposition(a_deg: float, b_deg: float) -> float:
    """1.0 when two headings are exactly opposed, 0.0 when aligned or square."""
    delta = math.radians(a_deg - b_deg)
    return max(0.0, -math.cos(delta))


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------


def _density_signal(density: float, band: Band, th: Thresholds) -> Signal:
    if density >= th.band_very_crowded:
        level: RiskLevel = "red"
    elif density >= th.band_crowded:
        level = "amber"
    else:
        level = "green"
    return Signal(
        name="density",
        level=level,
        value=round(density, 2),
        detail=f"{density:.2f} p/m2 ({band.replace('_', ' ')})",
    )


def _trend_signal(density: float, rate: float, ttc: Optional[float], th: Thresholds) -> Signal:
    level: RiskLevel = "green"
    if rate >= th.trend_rising_rate and ttc is not None:
        if ttc <= th.ttc_red_s:
            level = "red"
        elif ttc <= th.ttc_amber_s:
            level = "amber"

    if rate >= th.trend_rising_rate:
        detail = f"rising {rate:+.2f} p/m2/min"
        if ttc is not None:
            detail += f", {ttc / 60:.0f} min to critical"
    elif rate <= -th.trend_rising_rate:
        detail = f"falling {rate:+.2f} p/m2/min"
    else:
        detail = "steady"
    return Signal(name="density_trend", level=level, value=round(rate, 3), detail=detail)


def _breakdown_signal(density: float, speed: float, baseline: float,
                      rate: float, th: Thresholds) -> Signal:
    """Rising density with falling speed -- the actual crush precursor.

    A queue forming is inflow exceeding outflow, which shows up as density
    climbing while movement slows. Both halves matter:

      falling speed alone   - a crowd standing to watch an aarti, perfectly safe
      rising density alone  - a filling but freely moving space, still fine

    Gating on the pair is what lets this fire while density is merely 'busy',
    which is where the warning time lives. The floor on density stops camera
    noise in an empty frame from reading as a stall.
    """
    drop = 0.0
    if baseline > 0.05:
        drop = max(0.0, (baseline - speed) / baseline)

    rising = rate >= th.trend_rising_rate
    qualifies = density >= th.breakdown_density or (
        rising and density >= th.breakdown_density * 0.4
    )

    level: RiskLevel = "green"
    if qualifies:
        stalled = speed <= th.slow_speed_ms
        severe_drop = drop >= th.breakdown_speed_drop * 1.6
        if (stalled or severe_drop) and density >= th.band_crowded:
            level = "red"
        elif drop >= th.breakdown_speed_drop and speed < th.free_speed_ms:
            level = "amber"
        elif stalled and density >= th.breakdown_density:
            level = "amber"

    if level == "green":
        detail = f"{speed:.2f} m/s, flowing"
    else:
        detail = f"{speed:.2f} m/s, down {drop * 100:.0f}% from {baseline:.2f} at {density:.1f} p/m2"
    return Signal(name="flow_breakdown", level=level, value=round(drop, 3), detail=detail)


def _pressure_signal(density: float, variance: float, th: Thresholds) -> Signal:
    """Density x velocity variance: a proxy for people being pushed about.

    Not a physical pressure in pascals and not presented as one. It is an
    index that rises when a dense crowd stops moving as one body.
    """
    value = density * variance
    level: RiskLevel = "green"
    if value >= th.pressure_red:
        level = "red"
    elif value >= th.pressure_amber:
        level = "amber"
    return Signal(
        name="pressure",
        level=level,
        value=round(value, 3),
        detail=f"index {value:.2f} (density {density:.1f} x variance {variance:.2f})",
    )


def _convergence_signal(score: float, against: str, th: Thresholds) -> Signal:
    level: RiskLevel = "green"
    if score >= th.convergence_red:
        level = "red"
    elif score >= th.convergence_amber:
        level = "amber"
    detail = "no opposing flow" if level == "green" else f"opposing flow with {against} ({score:.2f})"
    return Signal(name="convergence", level=level, value=round(score, 3), detail=detail)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def time_to_critical(density: float, rate_per_min: float, th: Thresholds) -> Optional[float]:
    """Seconds until this zone reaches the crush-risk band at the current rate."""
    if density >= th.band_critical:
        return 0.0
    if rate_per_min < th.trend_rising_rate:
        return None
    seconds = (th.band_critical - density) / (rate_per_min / 60.0)
    return float(min(seconds, 3600.0))


def assess_scene(
    histories: Mapping[str, Sequence[Observation]],
    adjacency: Mapping[str, Sequence[str]] | None = None,
    thresholds: Thresholds | None = None,
    names: Mapping[str, str] | None = None,
) -> dict[str, ZoneAssessment]:
    """Assess every zone. Pure: same inputs, same output, always."""
    th = thresholds or Thresholds()
    adjacency = adjacency or {}
    names = names or {}
    out: dict[str, ZoneAssessment] = {}

    latest = {zid: h[-1] for zid, h in histories.items() if h}

    for zone_id, history in histories.items():
        if not history:
            out[zone_id] = ZoneAssessment()
            continue

        now = history[-1]
        window = _window(history, th.trend_window_s)
        band = band_for(now.density, th)

        rate = slope_per_min(window)
        ttc = time_to_critical(now.density, rate, th)
        baseline = speed_baseline(window)

        # Convergence looks outward, at the neighbours this zone was declared
        # adjacent to during scene setup.
        conv_score, conv_against = 0.0, ""
        for nb_id in adjacency.get(zone_id, ()):
            nb = latest.get(nb_id)
            if nb is None or now.flow_dir_deg is None or nb.flow_dir_deg is None:
                continue
            if now.mean_speed_ms < 0.1 or nb.mean_speed_ms < 0.1:
                continue
            opposition = _angle_opposition(now.flow_dir_deg, nb.flow_dir_deg)
            weight = min(1.0, min(now.density, nb.density) / max(th.band_crowded, 1e-6))
            score = opposition * weight
            if score > conv_score:
                conv_score, conv_against = score, names.get(nb_id, nb_id)

        signals = [
            _density_signal(now.density, band, th),
            _trend_signal(now.density, rate, ttc, th),
            _breakdown_signal(now.density, now.mean_speed_ms, baseline, rate, th),
            _convergence_signal(conv_score, conv_against, th),
            _pressure_signal(now.density, now.speed_variance, th),
        ]

        risk = worst(*(s.level for s in signals))
        by_name = {s.name: s for s in signals}
        headline = ""
        for candidate in HEADLINE_PRIORITY:
            sig = by_name.get(candidate)
            if sig is not None and sig.level == risk and risk != "green":
                headline = candidate
                break

        if rate >= th.trend_rising_rate:
            trend: Trend = "rising"
        elif rate <= -th.trend_rising_rate:
            trend = "falling"
        else:
            trend = "steady"

        out[zone_id] = ZoneAssessment(
            band=band,
            risk=risk,
            trend=trend,
            trend_rate_per_min=round(rate, 3),
            time_to_critical_s=None if ttc is None else round(ttc, 1),
            pressure=round(now.density * now.speed_variance, 3),
            convergence=round(conv_score, 3),
            headline=headline,
            action=SUGGESTED_ACTION.get(headline, ""),
            signals=signals,
        )

    return out
