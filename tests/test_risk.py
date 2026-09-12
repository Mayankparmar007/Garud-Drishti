"""Risk engine tests.

No public dataset contains drone footage from five minutes before a crowd
crush, so the risk logic cannot be validated against ground truth. What it
*can* be is fully specified and stress-tested against synthetic inputs, which
is what this file is. Each test states the scenario it encodes.
"""

from __future__ import annotations

import pytest

from uksi.contract import Thresholds
from uksi.worker.risk import Observation, assess_scene, slope_per_min, time_to_critical

TH = Thresholds()


def series(densities, speeds=None, start=1000.0, dt=1.0, variance=0.0, direction=90.0):
    speeds = speeds if speeds is not None else [1.0] * len(densities)
    return [
        Observation(
            ts=start + i * dt,
            density=d,
            count=d * 100,
            mean_speed_ms=s,
            flow_dir_deg=direction,
            speed_variance=variance,
        )
        for i, (d, s) in enumerate(zip(densities, speeds))
    ]


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Trend maths
# ---------------------------------------------------------------------------


def test_slope_of_flat_series_is_zero():
    assert slope_per_min(series([2.0] * 60)) == pytest.approx(0.0, abs=1e-9)


def test_slope_is_reported_per_minute():
    # 1.0 -> 2.0 over 60 samples at 1 s = +1.0 p/m2 per minute
    assert slope_per_min(series(ramp(1.0, 2.0, 60))) == pytest.approx(1.0, rel=0.05)


def test_slope_needs_enough_time_span():
    """Two seconds of a noisy signal is not a trend."""
    assert slope_per_min(series([1.0, 4.0, 1.0], dt=0.3)) == 0.0


def test_slope_survives_single_frame_spike():
    clean = ramp(2.0, 3.0, 60)
    noisy = list(clean)
    noisy[30] += 3.0  # one bad inference
    assert slope_per_min(series(noisy)) == pytest.approx(slope_per_min(series(clean)), abs=0.25)


def test_time_to_critical_extrapolates_to_the_crush_band():
    # at 3.0 p/m2 rising 1.0 p/m2/min, critical (5.0) is 2 minutes away
    assert time_to_critical(3.0, 1.0, TH) == pytest.approx(120.0)


def test_time_to_critical_is_none_when_not_rising():
    assert time_to_critical(3.0, 0.0, TH) is None
    assert time_to_critical(3.0, -0.5, TH) is None


def test_time_to_critical_is_zero_once_already_critical():
    assert time_to_critical(5.5, 1.0, TH) == 0.0


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_calm_scene_is_green_and_silent():
    out = assess_scene({"z1": series([0.8] * 60)})["z1"]
    assert out.risk == "green"
    assert out.band == "free"
    assert out.trend == "steady"
    assert out.headline == ""
    assert out.time_to_critical_s is None


def test_high_density_alone_reaches_red():
    out = assess_scene({"z1": series([4.5] * 60)})["z1"]
    assert out.risk == "red"
    assert out.band == "very_crowded"


def test_dense_but_freely_moving_crowd_is_not_a_breakdown():
    """A packed crowd walking at a normal pace is stable. Density says amber,
    but the flow-breakdown signal must stay green."""
    out = assess_scene({"z1": series([3.2] * 60, speeds=[1.1] * 60)})["z1"]
    sig = {s.name: s for s in out.signals}
    assert sig["flow_breakdown"].level == "green"
    assert sig["density"].level == "amber"


def test_sparse_crowd_standing_still_is_not_a_breakdown():
    """People watching an aarti are stationary and perfectly safe."""
    out = assess_scene({"z1": series([1.2] * 60, speeds=[0.05] * 60)})["z1"]
    assert out.risk == "green"


def test_flow_breakdown_fires_when_dense_crowd_stalls():
    """The money signal: density holding, speed collapsing."""
    speeds = [1.2] * 30 + list(ramp(1.2, 0.15, 30))
    out = assess_scene({"z1": series([3.3] * 60, speeds=speeds)})["z1"]
    sig = {s.name: s for s in out.signals}
    assert sig["flow_breakdown"].level == "red"
    assert out.risk == "red"
    assert out.headline == "flow_breakdown"
    assert "Stop inflow" in out.action


def test_breakdown_outranks_density_in_the_headline():
    """Both red, but the operator needs to hear the movement story."""
    speeds = [1.2] * 30 + [0.1] * 30
    out = assess_scene({"z1": series([4.6] * 60, speeds=speeds)})["z1"]
    assert out.risk == "red"
    assert out.headline == "flow_breakdown"


def test_breakdown_fires_while_density_is_still_only_busy():
    """The beat the whole demo is built on.

    Density is 2.2 p/m2 -- 'busy', nothing a human watching the video would
    call alarming -- but it is climbing while the crowd has lost half its
    walking speed. That is a queue forming, and it is the warning window.
    """
    out = assess_scene(
        {"z1": series(ramp(1.0, 2.2, 60), speeds=[1.2] * 25 + ramp(1.2, 0.5, 35))}
    )["z1"]
    sig = {s.name: s for s in out.signals}
    assert out.band == "busy"
    assert sig["density"].level == "green"  # raw density says nothing yet
    assert sig["flow_breakdown"].level == "amber"  # movement already has
    assert out.risk == "amber"
    assert out.headline == "flow_breakdown"


def test_rising_density_with_unchanged_speed_is_not_a_breakdown():
    """A space filling up while everyone still walks freely is just busy."""
    out = assess_scene({"z1": series(ramp(1.0, 2.5, 60), speeds=[1.15] * 60)})["z1"]
    sig = {s.name: s for s in out.signals}
    assert sig["flow_breakdown"].level == "green"


def test_sustained_jam_stays_red_after_the_speed_baseline_decays():
    """Once the whole window is slow there is no 'drop' left to measure.

    A jam must not quietly clear itself from the display just because it has
    been going on long enough to become the new normal.
    """
    out = assess_scene({"z1": series([4.2] * 60, speeds=[0.33] * 60)})["z1"]
    sig = {s.name: s for s in out.signals}
    assert sig["flow_breakdown"].value == pytest.approx(0.0)  # no measurable drop
    assert sig["flow_breakdown"].level == "red"  # still red on stalled + dense
    assert out.risk == "red"


def test_rising_trend_warns_before_the_threshold_is_crossed():
    """The whole pitch: amber while density is still merely 'busy'."""
    out = assess_scene({"z1": series(ramp(1.5, 2.6, 60))})["z1"]
    assert out.trend == "rising"
    assert out.trend_rate_per_min > 0.5
    assert out.band == "busy"  # not yet crowded
    assert out.risk == "amber"  # but already warning
    assert out.headline == "density_trend"
    assert out.time_to_critical_s is not None and out.time_to_critical_s < TH.ttc_amber_s


def test_falling_density_is_labelled_falling_and_never_gets_a_ttc():
    out = assess_scene({"z1": series(ramp(3.0, 1.5, 60))})["z1"]
    assert out.trend == "falling"
    assert out.time_to_critical_s is None


def test_convergence_detects_two_crowds_walking_into_each_other():
    hist = {
        "z1": series([3.1] * 60, speeds=[0.8] * 60, direction=0.0),
        "z2": series([3.1] * 60, speeds=[0.8] * 60, direction=180.0),
    }
    out = assess_scene(hist, adjacency={"z1": ["z2"], "z2": ["z1"]}, names={"z2": "Ramp East"})
    assert out["z1"].convergence > TH.convergence_red
    sig = {s.name: s for s in out["z1"].signals}
    assert sig["convergence"].level == "red"
    assert "Ramp East" in sig["convergence"].detail


def test_parallel_flow_between_neighbours_is_not_convergence():
    hist = {
        "z1": series([3.1] * 60, speeds=[0.8] * 60, direction=90.0),
        "z2": series([3.1] * 60, speeds=[0.8] * 60, direction=90.0),
    }
    out = assess_scene(hist, adjacency={"z1": ["z2"]})
    assert out["z1"].convergence == pytest.approx(0.0)


def test_convergence_in_a_sparse_crowd_is_weighted_down():
    """Two thin streams crossing is normal pedestrian behaviour."""
    hist = {
        "z1": series([0.4] * 60, speeds=[0.8] * 60, direction=0.0),
        "z2": series([0.4] * 60, speeds=[0.8] * 60, direction=180.0),
    }
    out = assess_scene(hist, adjacency={"z1": ["z2"]})
    assert out["z1"].convergence < TH.convergence_amber
    assert out["z1"].risk == "green"


def test_convergence_ignores_a_stationary_neighbour():
    """No flow direction to oppose if nobody is moving."""
    hist = {
        "z1": series([3.1] * 60, speeds=[0.8] * 60, direction=0.0),
        "z2": series([3.1] * 60, speeds=[0.02] * 60, direction=180.0),
    }
    out = assess_scene(hist, adjacency={"z1": ["z2"]})
    assert out["z1"].convergence == pytest.approx(0.0)


def test_pressure_index_rises_with_turbulent_movement():
    calm = assess_scene({"z1": series([3.0] * 60, variance=0.02)})["z1"]
    churn = assess_scene({"z1": series([3.0] * 60, variance=0.45)})["z1"]
    assert churn.pressure > calm.pressure
    sig = {s.name: s for s in churn.signals}
    assert sig["pressure"].level in ("amber", "red")


# ---------------------------------------------------------------------------
# Contract guarantees
# ---------------------------------------------------------------------------


def test_engine_is_pure():
    hist = {"z1": series(ramp(1.0, 3.0, 60))}
    a = assess_scene(hist)["z1"]
    b = assess_scene(hist)["z1"]
    assert (a.risk, a.trend_rate_per_min, a.time_to_critical_s) == (
        b.risk,
        b.trend_rate_per_min,
        b.time_to_critical_s,
    )


def test_empty_and_short_history_do_not_crash():
    out = assess_scene({"z1": [], "z2": series([2.0])})
    assert out["z1"].risk == "green"
    assert out["z2"].risk == "green"


def test_every_zone_always_reports_all_five_signals():
    out = assess_scene({"z1": series([2.0] * 30)})["z1"]
    assert {s.name for s in out.signals} == {
        "density",
        "density_trend",
        "flow_breakdown",
        "convergence",
        "pressure",
    }


def test_thresholds_are_tunable_without_touching_the_engine():
    """Site calibration must be a config change, not a code change."""
    strict = Thresholds(band_crowded=1.0, band_very_crowded=1.5)
    out = assess_scene({"z1": series([1.8] * 60)}, thresholds=strict)["z1"]
    assert out.risk == "red"
    assert assess_scene({"z1": series([1.8] * 60)})["z1"].risk == "green"
