"""Alert manager tests.

The alert rules exist to protect operator attention, so they are worth testing
as carefully as the maths. Every test below is a claim about what an operator
will and will not be interrupted by.
"""

from __future__ import annotations

import pytest

from uksi.contract import Signal, Thresholds
from uksi.worker.alerts import AlertManager
from uksi.worker.risk import ZoneAssessment

TH = Thresholds()  # dwell 25 s, resolve 45 s, reassert gap 120 s
NAMES = {"z1": "Gate Approach", "z2": "Main Ghat"}


def assessment(risk="amber", headline="density_trend", ttc=240.0, detail="rising 0.50 p/m2/min"):
    return ZoneAssessment(
        risk=risk,
        headline=headline if risk != "green" else "",
        time_to_critical_s=ttc,
        action="Slow inflow now" if risk != "green" else "",
        signals=[Signal(name=headline, level=risk, value=0.5, detail=detail)],
    )


GREEN = assessment(risk="green")


def run(mgr, seconds, assessment_at, start=1000.0):
    """Feed one sample per second and collect every event."""
    events = []
    for i in range(seconds):
        t = start + i
        _, evs = mgr.update(t, {"z1": assessment_at(i)}, NAMES)
        events.extend((t - start, a, kind) for a, kind in evs)
    return events


# ---------------------------------------------------------------------------
# Dwell
# ---------------------------------------------------------------------------


def test_nothing_fires_before_the_dwell_time():
    mgr = AlertManager(TH)
    events = run(mgr, 24, lambda i: assessment())
    assert events == []
    assert mgr.live() == []


def test_fires_once_the_dwell_time_is_satisfied():
    mgr = AlertManager(TH)
    events = run(mgr, 30, lambda i: assessment())
    assert [(t, kind) for t, _, kind in events] == [(25.0, "fired")]
    alert = mgr.live()[0]
    assert alert.severity == "amber"
    assert alert.zone_name == "Gate Approach"
    assert alert.metric == "density_trend"


def test_a_brief_spike_never_fires():
    """The anti-flicker guarantee. Ten seconds of amber is noise."""
    mgr = AlertManager(TH)
    events = run(mgr, 60, lambda i: assessment() if i < 10 else GREEN)
    assert events == []


def test_an_interrupted_condition_restarts_the_dwell():
    """20 s amber, one green sample, 20 s amber -- still nothing."""
    mgr = AlertManager(TH)
    events = run(mgr, 41, lambda i: GREEN if i == 20 else assessment())
    assert events == []


# ---------------------------------------------------------------------------
# Dedupe and escalation
# ---------------------------------------------------------------------------


def test_one_alert_per_zone_however_long_it_lasts():
    mgr = AlertManager(TH)
    events = run(mgr, 600, lambda i: assessment())
    assert len([e for e in events if e[2] == "fired"]) == 1
    assert len(mgr.live()) == 1


def test_worsening_escalates_in_place_and_keeps_the_id():
    mgr = AlertManager(TH)
    events = run(mgr, 60, lambda i: assessment() if i < 30 else assessment(risk="red"))
    kinds = [kind for _, _, kind in events]
    assert kinds == ["fired", "escalated"]
    assert events[0][1].id == events[1][1].id
    live = mgr.live()[0]
    assert live.severity == "red"
    assert live.escalations == 1
    assert len(mgr.live()) == 1


def test_escalation_needs_its_own_dwell():
    """A single red sample does not promote an amber."""
    mgr = AlertManager(TH)
    events = run(mgr, 40, lambda i: assessment(risk="red") if i in (30, 31) else assessment())
    assert [kind for _, _, kind in events] == ["fired"]
    assert mgr.live()[0].severity == "amber"


def test_an_alert_does_not_quietly_downgrade():
    """It records the worst the zone reached until it genuinely clears."""
    mgr = AlertManager(TH)
    run(mgr, 40, lambda i: assessment(risk="red"))
    assert mgr.live()[0].severity == "red"
    mgr.update(1100.0, {"z1": assessment(risk="amber")}, NAMES)
    assert mgr.live()[0].severity == "red"


# ---------------------------------------------------------------------------
# Resolution and hysteresis
# ---------------------------------------------------------------------------


def test_resolves_after_sustained_clear():
    mgr = AlertManager(TH)
    events = run(mgr, 80, lambda i: assessment() if i < 30 else GREEN)
    assert [kind for _, _, kind in events] == ["fired", "resolved"]
    resolved = events[-1][1]
    assert resolved.state == "resolved"
    assert resolved.resolved_at is not None
    assert mgr.live() == []


def test_one_non_green_sample_restarts_the_resolve_timer():
    """A zone that keeps twitching stays on the board."""
    mgr = AlertManager(TH)
    events = run(mgr, 120, lambda i: assessment() if i < 30 or i % 20 == 0 else GREEN)
    assert [kind for _, _, kind in events] == ["fired"]
    assert len(mgr.live()) == 1


def test_a_resolved_zone_is_held_back_from_re_firing_amber():
    mgr = AlertManager(TH)

    def pattern(i):
        if i < 30:
            return assessment()
        if i < 80:
            return GREEN  # resolves at i = 75
        return assessment()

    events = run(mgr, 130, pattern)
    kinds = [kind for _, _, kind in events]
    assert kinds == ["fired", "resolved"]  # the second amber is suppressed
    assert mgr.live() == []


def test_red_overrides_the_re_assert_hold():
    """Suppressing a red to keep the feed tidy is indefensible."""
    mgr = AlertManager(TH)

    def pattern(i):
        if i < 30:
            return assessment()
        if i < 80:
            return GREEN
        return assessment(risk="red")

    events = run(mgr, 130, pattern)
    kinds = [kind for _, _, kind in events]
    assert kinds == ["fired", "resolved", "fired"]
    assert mgr.live()[0].severity == "red"


# ---------------------------------------------------------------------------
# Zones are independent
# ---------------------------------------------------------------------------


def test_zones_are_tracked_separately():
    mgr = AlertManager(TH)
    for i in range(30):
        mgr.update(1000.0 + i, {"z1": assessment(), "z2": GREEN}, NAMES)
    assert [a.zone for a in mgr.live()] == ["z1"]

    for i in range(30, 60):
        mgr.update(1000.0 + i, {"z1": assessment(), "z2": assessment(risk="red")}, NAMES)
    assert [a.zone for a in mgr.live()] == ["z2", "z1"]  # red first


# ---------------------------------------------------------------------------
# Evidence and messages
# ---------------------------------------------------------------------------


def test_evidence_is_captured_when_the_alert_fires():
    captured = []

    def evidence(alert_id, zone_id):
        captured.append((alert_id, zone_id))
        return f"/frames/{alert_id}.jpg"

    mgr = AlertManager(TH, evidence=evidence)
    run(mgr, 30, lambda i: assessment())
    assert captured == [("a1", "z1")]
    assert mgr.live()[0].thumb == "/frames/a1.jpg"


def test_a_failing_evidence_capture_does_not_lose_the_alert():
    def evidence(alert_id, zone_id):
        raise OSError("disk full")

    mgr = AlertManager(TH, evidence=evidence)
    run(mgr, 30, lambda i: assessment())
    assert len(mgr.live()) == 1
    assert mgr.live()[0].thumb is None


def test_ids_continue_from_where_storage_left_off():
    """Restarting the worker must not reuse an id already in the database."""
    mgr = AlertManager(TH, start_seq=6)
    run(mgr, 30, lambda i: assessment())
    assert mgr.live()[0].id == "a7"


def test_trend_message_states_the_lead_time():
    mgr = AlertManager(TH)
    run(mgr, 30, lambda i: assessment(headline="density_trend", ttc=240.0))
    assert mgr.live()[0].message == "Density rising - 4 min to critical"


def test_breakdown_message_carries_the_movement_detail():
    mgr = AlertManager(TH)
    run(mgr, 30, lambda i: assessment(
        risk="red", headline="flow_breakdown", detail="0.22 m/s, down 64% from 1.20 at 3.1 p/m2"
    ))
    msg = mgr.live()[0].message
    assert msg.startswith("Movement breaking down")
    assert "0.22 m/s" in msg


def test_alert_carries_a_suggested_action():
    mgr = AlertManager(TH)
    run(mgr, 30, lambda i: assessment())
    assert mgr.live()[0].action == "Slow inflow now"


# ---------------------------------------------------------------------------
# The rate claim
# ---------------------------------------------------------------------------


def test_a_realistic_noisy_hour_stays_under_five_alerts():
    """The design target. Density hovering at the amber edge with model noise
    must not produce a stream of interruptions."""
    import random

    rng = random.Random(11)
    mgr = AlertManager(TH)
    fired = 0
    for i in range(3600):
        # Amber roughly a third of the time, in short bursts, as noise would.
        risk = "amber" if rng.random() < 0.35 else "green"
        _, evs = mgr.update(1000.0 + i, {"z1": assessment(risk=risk)}, NAMES)
        fired += sum(1 for _, kind in evs if kind == "fired")
    assert fired <= 5
