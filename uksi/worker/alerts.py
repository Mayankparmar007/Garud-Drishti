"""Alert manager.

A feed that spams gets ignored, and an ignored feed is worse than no feed --
it costs the operator attention and returns nothing. So the rules are strict:

  dwell       a condition holds continuously for ~25 s before anything fires
  dedupe      one live alert per zone, ever
  escalate    a worsening zone updates its alert rather than adding another
  hysteresis  it must be clear for ~45 s to resolve, and cannot immediately
              re-fire afterwards

No clock, no disk, no database. Time is a parameter and evidence capture is a
callback, so the whole state machine is testable second by second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

from ..contract import Alert, RISK_ORDER, RiskLevel, Thresholds
from .risk import SIGNAL_LABELS, SUGGESTED_ACTION, ZoneAssessment

TITLES = {
    "density": "High density",
    "density_trend": "Density rising",
    "flow_breakdown": "Movement breaking down",
    "convergence": "Opposing flows meeting",
    "pressure": "Turbulent movement",
}

EventKind = str  # "fired" | "escalated" | "resolved"


def _message(assessment: ZoneAssessment) -> str:
    headline = assessment.headline or "density"
    detail = next((s.detail for s in assessment.signals if s.name == headline), "")
    title = TITLES.get(headline, SIGNAL_LABELS.get(headline, headline))

    if headline == "density_trend" and assessment.time_to_critical_s is not None:
        mins = assessment.time_to_critical_s / 60.0
        lead = f"{mins:.0f} min" if mins >= 1 else f"{assessment.time_to_critical_s:.0f} s"
        return f"{title} - {lead} to critical"
    return f"{title} - {detail}" if detail else title


@dataclass
class ZoneAlertState:
    pending_since: Optional[float] = None
    active: Optional[Alert] = None
    clear_since: Optional[float] = None
    last_resolved_at: Optional[float] = None


@dataclass
class AlertManager:
    thresholds: Thresholds = field(default_factory=Thresholds)
    evidence: Optional[Callable[[str, str], Optional[str]]] = None
    start_seq: int = 0

    def __post_init__(self) -> None:
        self._seq = self.start_seq
        self._states: dict[str, ZoneAlertState] = {}

    def _next_id(self) -> str:
        self._seq += 1
        return f"a{self._seq}"

    def _capture(self, alert_id: str, zone_id: str) -> Optional[str]:
        if self.evidence is None:
            return None
        try:
            return self.evidence(alert_id, zone_id)
        except Exception:
            return None  # evidence is nice to have; never break the alert for it

    def update(
        self,
        now: float,
        assessments: Mapping[str, ZoneAssessment],
        names: Mapping[str, str] | None = None,
    ) -> tuple[list[Alert], list[tuple[Alert, EventKind]]]:
        """Advance the machine one sample. Returns (live alerts, events)."""
        th = self.thresholds
        names = names or {}
        events: list[tuple[Alert, EventKind]] = []
        ts = datetime.fromtimestamp(now, tz=timezone.utc)

        for zone_id, assessment in assessments.items():
            st = self._states.setdefault(zone_id, ZoneAlertState())
            level: RiskLevel = assessment.risk

            if level == "green":
                st.pending_since = None
                if st.active is not None:
                    if st.clear_since is None:
                        st.clear_since = now
                    elif now - st.clear_since >= th.resolve_s:
                        alert = st.active
                        alert.state = "resolved"
                        alert.resolved_at = ts
                        st.active, st.clear_since = None, None
                        st.last_resolved_at = now
                        events.append((alert, "resolved"))
                continue

            st.clear_since = None  # any non-green sample restarts the resolve timer

            if st.active is not None:
                # Already reported. Escalate only on a genuine worsening, and
                # give that its own shorter dwell so a single red sample does
                # not promote an amber.
                if RISK_ORDER[level] > RISK_ORDER[st.active.severity]:
                    if st.pending_since is None:
                        st.pending_since = now
                    elif now - st.pending_since >= th.dwell_s * 0.4:
                        alert = st.active
                        alert.severity = level
                        alert.metric = assessment.headline or alert.metric
                        alert.message = _message(assessment)
                        alert.action = assessment.action or alert.action
                        alert.escalations += 1
                        alert.ts = ts
                        alert.thumb = self._capture(alert.id, zone_id) or alert.thumb
                        st.pending_since = None
                        events.append((alert, "escalated"))
                else:
                    st.pending_since = None
                continue

            # Nothing live for this zone yet.
            if st.pending_since is None:
                st.pending_since = now
                continue
            if now - st.pending_since < th.dwell_s:
                continue

            # A zone that just resolved is held back briefly so a borderline
            # condition cannot oscillate the feed. Red overrides the hold:
            # suppressing a red to keep the display tidy is indefensible.
            if (
                st.last_resolved_at is not None
                and now - st.last_resolved_at < th.reassert_gap_s
                and level != "red"
            ):
                continue

            alert_id = self._next_id()
            alert = Alert(
                id=alert_id,
                zone=zone_id,
                zone_name=names.get(zone_id, zone_id),
                severity=level,
                metric=assessment.headline or "density",
                message=_message(assessment),
                action=assessment.action or SUGGESTED_ACTION.get(assessment.headline, ""),
                thumb=self._capture(alert_id, zone_id),
                ts=ts,
                first_seen=datetime.fromtimestamp(st.pending_since, tz=timezone.utc),
            )
            st.active = alert
            st.pending_since = None
            events.append((alert, "fired"))

        return self.live(), events

    def live(self) -> list[Alert]:
        """Active alerts, most severe first, then most recent."""
        out = [s.active for s in self._states.values() if s.active is not None]
        out.sort(key=lambda a: (-RISK_ORDER[a.severity], -a.ts.timestamp()))
        return out
