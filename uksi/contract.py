"""The data contract.

Frozen first, on purpose: the worker, the API, the mock generator and the
frontend all read this file's shape, so backend and frontend never block each
other. Anything added later must be additive.

Two documents:

  SceneConfig  - what an operator sets up once, before the event. Persisted.
  StatePayload - what the control room receives, once a second. Ephemeral.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Point = tuple[float, float]

RiskLevel = Literal["green", "amber", "red"]
Band = Literal["free", "busy", "crowded", "very_crowded", "critical"]
Trend = Literal["rising", "steady", "falling"]
CorridorStatus = Literal["clear", "restricted", "blocked"]
SourceKind = Literal["rtsp", "file", "synthetic"]

BAND_ORDER: tuple[Band, ...] = ("free", "busy", "crowded", "very_crowded", "critical")
BAND_LABELS: dict[str, str] = {
    "free": "Free flow",
    "busy": "Busy",
    "crowded": "Crowded",
    "very_crowded": "Very crowded",
    "critical": "Crush risk",
}
RISK_ORDER: dict[str, int] = {"green": 0, "amber": 1, "red": 2}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------


class Thresholds(BaseModel):
    """Every number the risk engine can be tuned by.

    Defaults are the published density bands plus first-guess movement
    thresholds. The movement ones are expected to be calibrated against the
    actual scene -- a wide ghat and a narrow lane do not break down at the
    same speed.
    """

    # Density bands, persons per square metre.
    band_busy: float = 2.0
    band_crowded: float = 3.0
    band_very_crowded: float = 4.0
    band_critical: float = 5.0

    # Trend. Slope is measured over trend_window_s and reported per minute.
    trend_window_s: float = 60.0
    trend_rising_rate: float = 0.12  # p/m2/min above which we call it rising
    ttc_amber_s: float = 300.0
    ttc_red_s: float = 120.0

    # Movement. free_speed is unimpeded walking, slow_speed is shuffling.
    free_speed_ms: float = 1.0
    slow_speed_ms: float = 0.35
    breakdown_density: float = 2.0  # below this, slow movement is just loitering
    breakdown_speed_drop: float = 0.35  # fractional fall from window baseline

    # Pressure proxy: density x velocity variance.
    pressure_amber: float = 0.60
    pressure_red: float = 1.20

    # Convergence between neighbouring zones, 0-1.
    convergence_amber: float = 0.45
    convergence_red: float = 0.70

    # Corridors.
    corridor_clear: float = 1.0
    corridor_restricted: float = 2.0

    # Alert state machine, seconds.
    dwell_s: float = 25.0
    resolve_s: float = 45.0
    reassert_gap_s: float = 120.0


class Calibration(BaseModel):
    """Four image points and their real ground coordinates, in metres.

    The operator clicks a shape on the ground whose dimensions are known --
    a road width, a barrier span, something measurable on satellite view --
    and types the dimensions in. That is the entire calibration.
    """

    image_points: list[Point] = Field(default_factory=list)
    world_points: list[Point] = Field(default_factory=list)
    note: str = ""

    @field_validator("image_points", "world_points")
    @classmethod
    def _four_or_none(cls, v: list[Point]) -> list[Point]:
        if v and len(v) != 4:
            raise ValueError("calibration needs exactly 4 points")
        return v

    @property
    def is_set(self) -> bool:
        return len(self.image_points) == 4 and len(self.world_points) == 4


class ZoneConfig(BaseModel):
    id: str
    name: str
    polygon: list[Point]  # image pixels, in frame coordinates
    neighbors: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("polygon")
    @classmethod
    def _at_least_triangle(cls, v: list[Point]) -> list[Point]:
        if len(v) < 3:
            raise ValueError("a zone needs at least 3 points")
        return v


class CorridorConfig(BaseModel):
    """An evacuation route, drawn by hand before the event.

    A real operator marks corridors in the briefing anyway. We report density
    along them rather than trying to discover them.
    """

    id: str
    name: str
    polygon: list[Point]
    width_m: Optional[float] = None
    note: str = ""

    @field_validator("polygon")
    @classmethod
    def _at_least_triangle(cls, v: list[Point]) -> list[Point]:
        if len(v) < 3:
            raise ValueError("a corridor needs at least 3 points")
        return v


class SourceConfig(BaseModel):
    kind: SourceKind = "synthetic"
    uri: str = ""


class SceneConfig(BaseModel):
    name: str = "Unconfigured scene"
    source: SourceConfig = Field(default_factory=SourceConfig)
    frame_size: Optional[tuple[int, int]] = None  # (w, h) the geometry was drawn against
    calibration: Calibration = Field(default_factory=Calibration)
    zones: list[ZoneConfig] = Field(default_factory=list)
    corridors: list[CorridorConfig] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # Site calibration for the counting model: the factor that makes reported
    # counts match a hand count on one frame. Pretrained crowd models are
    # trained at ground-level scales and systematically mis-count from
    # altitude; this is the honest one-knob correction, and it is recorded in
    # config rather than buried in the model wrapper.
    count_scale: float = 1.0
    # Which density backend suits this scene: auto | csrnet | yolo | blob.
    # "auto" walks the ladder in uksi.worker.density and takes the first that
    # loads, which is right for real footage. It is wrong for the simulator:
    # a person detector loads perfectly happily and then finds nothing at all
    # in a rendered scene, so it never degrades to the backend that works. The
    # scene knows what it is; it says so here.
    density_backend: Literal["auto", "csrnet", "yolo", "blob"] = "auto"
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def calibrated(self) -> bool:
        return self.calibration.is_set


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """One reason the risk engine reached its verdict.

    Carried all the way to the UI so an operator can see *why* a zone went
    amber, not just that it did.
    """

    name: str
    level: RiskLevel
    value: float
    detail: str


class ZoneState(BaseModel):
    id: str
    name: str
    polygon: list[Point]
    count: float
    area_m2: float
    density: float
    band: Band
    risk: RiskLevel
    trend: Trend
    trend_rate_per_min: float
    time_to_critical_s: Optional[float] = None
    mean_speed_ms: float = 0.0
    flow_dir_deg: Optional[float] = None
    speed_variance: float = 0.0
    pressure: float = 0.0
    convergence: float = 0.0
    headline: str = ""
    signals: list[Signal] = Field(default_factory=list)


class CorridorState(BaseModel):
    id: str
    name: str
    polygon: list[Point]
    status: CorridorStatus
    count: float
    area_m2: float
    density: float
    mean_speed_ms: float = 0.0
    flow_dir_deg: Optional[float] = None


class Alert(BaseModel):
    id: str
    zone: str
    zone_name: str
    severity: RiskLevel
    metric: str
    message: str
    action: str = ""
    thumb: Optional[str] = None
    ts: datetime
    first_seen: datetime
    state: Literal["active", "resolved"] = "active"
    escalations: int = 0
    resolved_at: Optional[datetime] = None


class SceneSummary(BaseModel):
    name: str
    calibrated: bool
    frame_w: int = 0
    frame_h: int = 0
    zone_count: int = 0


class SourceStatus(BaseModel):
    kind: SourceKind
    uri: str = ""
    connected: bool = False
    detail: str = ""


class Totals(BaseModel):
    count: float = 0.0
    area_m2: float = 0.0
    density: float = 0.0
    zones_amber: int = 0
    zones_red: int = 0


class StatePayload(BaseModel):
    """Exactly what goes over the websocket, once a second."""

    ts: datetime
    seq: int = 0
    latency_ms: int = 0
    proc_ms: int = 0
    fps: float = 0.0
    frame_url: str = "/frames/latest.jpg"
    heat_url: Optional[str] = None
    heat_max_density: float = 0.0
    model: str = ""
    source: SourceStatus
    scene: SceneSummary
    totals: Totals = Field(default_factory=Totals)
    zones: list[ZoneState] = Field(default_factory=list)
    corridors: list[CorridorState] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    # Present only when the source knows the true answer -- i.e. the simulator.
    # It carries the counting error, so accuracy is visible during a run
    # instead of being asserted afterwards. Always null on real footage.
    debug: Optional[dict] = None

    def to_wire(self) -> dict:
        return self.model_dump(mode="json")


def band_for(density: float, th: Thresholds) -> Band:
    if density >= th.band_critical:
        return "critical"
    if density >= th.band_very_crowded:
        return "very_crowded"
    if density >= th.band_crowded:
        return "crowded"
    if density >= th.band_busy:
        return "busy"
    return "free"


def worst(*levels: RiskLevel) -> RiskLevel:
    out: RiskLevel = "green"
    for lv in levels:
        if RISK_ORDER[lv] > RISK_ORDER[out]:
            out = lv
    return out
