"""The processing worker.

Reads frames, measures, decides, publishes. Runs in its own thread so a slow
or absent dashboard cannot stall the analysis, and so the analysis cannot stall
the API.

Order of work per frame:

    read -> resize -> density -> per-zone counts -> flow -> risk -> alerts
         -> assemble state -> publish

Scene geometry is recompiled only when the configuration changes. Polygon
masks and ground areas cost real time to derive and never change between
edits, so they are never in the per-frame path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from . import settings
from .contract import (
    CorridorState,
    SceneConfig,
    SceneSummary,
    StatePayload,
    Totals,
    ZoneState,
)
from .db import Store
from .store import LiveState
from .worker.alerts import AlertManager
from .worker.calibration import Calibrator, polygon_mask
from .worker.density import DensityBackend, make_backend
from .worker.flow import FlowStage
from .worker.overlay import density_per_m2_map, encode_jpeg, render_heat_png
from .worker.risk import Observation, assess_scene

log = logging.getLogger(__name__)


@dataclass
class Region:
    """A zone or corridor, compiled against the working frame size."""

    id: str
    name: str
    polygon: list[tuple[float, float]]
    mask: np.ndarray
    area_m2: float
    neighbors: list[str] = field(default_factory=list)


@dataclass
class SceneRuntime:
    """Everything derived from a SceneConfig for one frame size."""

    config: SceneConfig
    work_size: tuple[int, int]
    calibrator: Calibrator
    zones: list[Region]
    corridors: list[Region]
    m2_per_px: np.ndarray

    @property
    def names(self) -> dict[str, str]:
        return {z.id: z.name for z in self.zones}

    @property
    def adjacency(self) -> dict[str, list[str]]:
        return {z.id: z.neighbors for z in self.zones}


def compile_scene(cfg: SceneConfig, work_w: int, work_h: int) -> SceneRuntime:
    """Turn a configuration into masks, areas and a calibrator.

    Geometry is stored against whatever frame size it was drawn on. If the
    stream resolution later changes, everything is rescaled rather than
    silently landing in the wrong place -- a zone that quietly shifts by a
    factor of 1.5 would produce plausible, wrong numbers.
    """
    src_w, src_h = cfg.frame_size or (work_w, work_h)
    sx = work_w / src_w if src_w else 1.0
    sy = work_h / src_h if src_h else 1.0

    def scale(poly) -> list[tuple[float, float]]:
        return [(float(x) * sx, float(y) * sy) for x, y in poly]

    if cfg.calibration.is_set:
        cal = Calibrator.from_points(scale(cfg.calibration.image_points), cfg.calibration.world_points)
    else:
        cal = Calibrator.uncalibrated()

    zones = []
    for z in cfg.zones:
        poly = scale(z.polygon)
        zones.append(
            Region(
                id=z.id, name=z.name, polygon=poly,
                mask=polygon_mask(poly, work_w, work_h),
                area_m2=cal.area_m2(poly),
                neighbors=list(z.neighbors),
            )
        )

    corridors = []
    for c in cfg.corridors:
        poly = scale(c.polygon)
        corridors.append(
            Region(
                id=c.id, name=c.name, polygon=poly,
                mask=polygon_mask(poly, work_w, work_h),
                area_m2=cal.area_m2(poly),
            )
        )

    return SceneRuntime(
        config=cfg,
        work_size=(work_w, work_h),
        calibrator=cal,
        zones=zones,
        corridors=corridors,
        m2_per_px=cal.pixel_areas(work_w, work_h),
    )


class Pipeline:
    def __init__(self, live: LiveState, store: Store, source_factory=None):
        self.live = live
        self.store = store
        self._source_factory = source_factory
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cfg_lock = threading.Lock()
        self._pending_cfg: Optional[SceneConfig] = None
        self._reset_requested = False
        self._cfg: Optional[SceneConfig] = None
        self._runtime: Optional[SceneRuntime] = None

        self.source = None
        self.backend: Optional[DensityBackend] = None
        self._backend_prefer: Optional[str] = None
        self.flow = FlowStage()
        self.alerts = AlertManager(start_seq=store.max_alert_seq())

        self._history: dict[str, list[Observation]] = {}
        self._current_frame: Optional[np.ndarray] = None
        self._current_zone_polys: dict[str, list] = {}
        self._seq = 0
        self._fps = 0.0
        self._last_heat = 0.0
        self._last_purge = 0.0

    # -- lifecycle --------------------------------------------------------

    def apply_scene(self, cfg: SceneConfig) -> None:
        """Hand a new configuration to the worker. Safe from any thread."""
        with self._cfg_lock:
            self._pending_cfg = cfg

    def start(self) -> "Pipeline":
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self.source:
            self.source.close()

    def reset_history(self) -> None:
        """Drop the measurement history and the flow reference frame.

        Called when the scene underneath the measurements has been moved --
        today only by the demo scenario control. Trend is a slope fitted over
        real elapsed time, so carrying samples across a discontinuity would
        manufacture a rate of change that nothing in the world produced.

        Requested rather than performed: history and the flow stage belong to
        the worker thread, and clearing a dict out from under it mid-frame is
        the kind of race that shows up once, on stage.
        """
        with self._cfg_lock:
            self._reset_requested = True

    # -- configuration ----------------------------------------------------

    def _take_pending(self) -> Optional[SceneConfig]:
        with self._cfg_lock:
            cfg, self._pending_cfg = self._pending_cfg, None
        return cfg

    def _take_reset(self) -> bool:
        with self._cfg_lock:
            requested, self._reset_requested = self._reset_requested, False
        return requested

    def _adopt(self, cfg: SceneConfig) -> None:
        source_changed = self._cfg is None or cfg.source != self._cfg.source
        self._cfg = cfg
        self.alerts.thresholds = cfg.thresholds
        self._runtime = None  # recompiled on the next frame, where the size is known
        self.flow.reset()
        self._history.clear()

        if source_changed:
            if self.source is not None:
                self.source.close()
            factory = self._source_factory
            if factory is None:
                from .worker.source import make_source as factory  # noqa: N813
            self.source = factory(cfg.source)
            log.info("source: %s %s", cfg.source.kind, cfg.source.uri)

        # Backend choice. The environment variable is a deployment override and
        # outranks everything; otherwise the scene picks, and "auto" walks the
        # ladder. Rebuilt only when the answer actually changes -- loading a
        # model takes seconds, and a scene edit should not stall the feed.
        prefer = settings.DENSITY_BACKEND
        if prefer == "auto":
            prefer = cfg.density_backend
        if (
            self.backend is None
            or prefer != self._backend_prefer
            or getattr(self.backend, "count_scale", 1.0) != cfg.count_scale
        ):
            self._backend_prefer = prefer
            self.backend = make_backend(
                prefer=prefer,
                csrnet_weights=settings.CSRNET_WEIGHTS,
                yolo_weights=settings.YOLO_WEIGHTS,
                count_scale=cfg.count_scale,
            )

    # -- evidence ---------------------------------------------------------

    def _capture_evidence(self, alert_id: str, zone_id: str) -> Optional[str]:
        """Write the downscaled evidence thumbnail for a fired alert."""
        frame = self._current_frame
        if frame is None:
            return None
        img = frame.copy()
        poly = self._current_zone_polys.get(zone_id)
        if poly:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (60, 75, 240), 2)

        settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.FRAMES_DIR / f"{alert_id}.jpg"
        data = encode_jpeg(img, long_edge=settings.THUMB_LONG_EDGE)
        if not data:
            return None
        path.write_bytes(data)
        return f"/frames/{alert_id}.jpg"

    # -- main loop --------------------------------------------------------

    def _run(self) -> None:
        min_interval = 1.0 / max(settings.SAMPLE_FPS, 0.1)
        prev_ts = 0.0
        last_start = 0.0

        while not self._stop.is_set():
            pending = self._take_pending()
            if pending is not None:
                self._adopt(pending)
            if self._take_reset():
                self._history.clear()
                self.flow.reset()
                prev_ts = 0.0  # the next dt would otherwise span the jump
                log.info("measurement history reset")
            if self.source is None:
                self._stop.wait(0.2)
                continue

            # Pace the loop. Reading faster than we can process only builds a
            # queue, and a queue is indistinguishable from latency.
            wait = min_interval - (time.time() - last_start)
            if wait > 0:
                self._stop.wait(wait)
            last_start = time.time()

            frame, capture_ts, fresh = self.source.read(timeout=0.5)
            if frame is None or not fresh:
                self._publish_idle()
                continue

            try:
                dt = capture_ts - prev_ts if prev_ts else 0.0
                self._process(frame, capture_ts, dt)
                prev_ts = capture_ts
            except Exception:
                log.exception("frame processing failed; continuing")

            self._maybe_purge()

    def _work_size(self, frame: np.ndarray) -> tuple[int, int]:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= settings.INFERENCE_LONG_EDGE:
            return w, h
        s = settings.INFERENCE_LONG_EDGE / longest
        return int(round(w * s)), int(round(h * s))

    def _process(self, raw: np.ndarray, capture_ts: float, dt: float) -> None:
        t0 = time.time()
        raw_h, raw_w = raw.shape[:2]
        work_w, work_h = self._work_size(raw)
        frame = raw if (raw_w, raw_h) == (work_w, work_h) else cv2.resize(
            raw, (work_w, work_h), interpolation=cv2.INTER_AREA
        )
        self._current_frame = frame

        cfg = self._cfg or SceneConfig()
        if self._runtime is None or self._runtime.work_size != (work_w, work_h):
            self._runtime = compile_scene(cfg, work_w, work_h)
            self._current_zone_polys = {z.id: z.polygon for z in self._runtime.zones}
            log.info(
                "scene compiled: %d zones, %d corridors, %s at %dx%d",
                len(self._runtime.zones), len(self._runtime.corridors),
                "calibrated" if self._runtime.calibrator.is_calibrated else "UNCALIBRATED",
                work_w, work_h,
            )
        rt = self._runtime

        density = self.backend.density(frame)
        flow_field = self.flow.flow_for(frame, density)
        now = time.time()

        zone_states: list[ZoneState] = []
        observations: dict[str, Observation] = {}

        for zone in rt.zones:
            count = float(density[zone.mask].sum())
            area = max(zone.area_m2, 1e-6)
            zf = self.flow.zone_flow(zone.id, flow_field, zone.mask, density, rt.calibrator, dt)
            obs = Observation(
                ts=capture_ts,
                density=count / area,
                count=count,
                mean_speed_ms=zf.mean_speed_ms,
                flow_dir_deg=zf.flow_dir_deg,
                speed_variance=zf.speed_variance,
            )
            observations[zone.id] = obs
            hist = self._history.setdefault(zone.id, [])
            hist.append(obs)
            if len(hist) > settings.HISTORY_SAMPLES:
                del hist[: len(hist) - settings.HISTORY_SAMPLES]

        assessments = assess_scene(
            {z.id: self._history.get(z.id, []) for z in rt.zones},
            adjacency=rt.adjacency,
            thresholds=cfg.thresholds,
            names=rt.names,
        )

        for zone in rt.zones:
            obs = observations[zone.id]
            a = assessments[zone.id]
            zone_states.append(
                ZoneState(
                    id=zone.id, name=zone.name, polygon=zone.polygon,
                    count=round(obs.count, 1), area_m2=round(zone.area_m2, 1),
                    density=round(obs.density, 2),
                    band=a.band, risk=a.risk, trend=a.trend,
                    trend_rate_per_min=a.trend_rate_per_min,
                    time_to_critical_s=a.time_to_critical_s,
                    mean_speed_ms=round(obs.mean_speed_ms, 2),
                    flow_dir_deg=None if obs.flow_dir_deg is None else round(obs.flow_dir_deg, 1),
                    speed_variance=round(obs.speed_variance, 3),
                    pressure=a.pressure, convergence=a.convergence,
                    headline=a.headline, signals=a.signals,
                )
            )

        # Alerts. Evidence capture happens inside update(), which is why the
        # current frame is set before the call.
        self.alerts.evidence = self._capture_evidence
        live_alerts, events = self.alerts.update(now, assessments, rt.names)
        for alert, kind in events:
            self.store.upsert_alert(alert)
            log.info("alert %s %s: %s [%s] %s", kind, alert.id, alert.zone_name,
                     alert.severity, alert.message)

        corridor_states = self._corridors(rt, density, flow_field, dt, cfg)

        # Heat map is throttled and rendered at reduced resolution: the field is
        # smoothed over about a metre of ground, so full resolution carries no
        # extra information and costs time on every frame.
        heat_max = 0.0
        if now - self._last_heat >= 1.0 / max(settings.PUBLISH_HZ, 0.1):
            field = density_per_m2_map(density, rt.m2_per_px, downscale=2)
            png, heat_max = render_heat_png(field, downscale=1)
            if png:
                self.live.set_heat(png)
            self._last_heat = now

        self.live.set_frame(encode_jpeg(frame, quality=76))

        proc_ms = int((time.time() - t0) * 1000)
        self._seq += 1
        alpha = 0.2
        inst_fps = 1.0 / dt if dt > 0.01 else 0.0
        self._fps = inst_fps if self._fps == 0 else (1 - alpha) * self._fps + alpha * inst_fps

        self.live.publish(
            StatePayload(
                ts=datetime.fromtimestamp(capture_ts, tz=timezone.utc),
                seq=self._seq,
                latency_ms=int((time.time() - capture_ts) * 1000),
                proc_ms=proc_ms,
                fps=round(self._fps, 2),
                heat_url="/frames/heat.png",
                heat_max_density=round(heat_max, 2),
                model=f"{self.backend.name} ({self.backend.detail})",
                source=self.source.status,
                scene=SceneSummary(
                    name=cfg.name,
                    calibrated=rt.calibrator.is_calibrated,
                    frame_w=work_w, frame_h=work_h,
                    zone_count=len(rt.zones),
                ),
                totals=self._totals(zone_states),
                zones=zone_states,
                corridors=corridor_states,
                alerts=live_alerts,
                debug=self._debug(density, (work_w / raw_w, work_h / raw_h)),
            )
        )

    def _corridors(self, rt: SceneRuntime, density, flow_field, dt, cfg) -> list[CorridorState]:
        th = cfg.thresholds
        out = []
        for c in rt.corridors:
            count = float(density[c.mask].sum())
            area = max(c.area_m2, 1e-6)
            d = count / area
            zf = self.flow.zone_flow(f"corridor:{c.id}", flow_field, c.mask, density,
                                     rt.calibrator, dt)

            # A corridor is only useful if people can move along it, so a route
            # that is merely busy but stalled counts as blocked, not restricted.
            if d >= th.corridor_restricted or (d >= th.corridor_clear and 0 < zf.mean_speed_ms < th.slow_speed_ms):
                status = "blocked"
            elif d >= th.corridor_clear:
                status = "restricted"
            else:
                status = "clear"

            out.append(
                CorridorState(
                    id=c.id, name=c.name, polygon=c.polygon, status=status,
                    count=round(count, 1), area_m2=round(c.area_m2, 1), density=round(d, 2),
                    mean_speed_ms=round(zf.mean_speed_ms, 2),
                    flow_dir_deg=None if zf.flow_dir_deg is None else round(zf.flow_dir_deg, 1),
                )
            )
        return out

    @staticmethod
    def _totals(zones: list[ZoneState]) -> Totals:
        count = sum(z.count for z in zones)
        area = sum(z.area_m2 for z in zones)
        return Totals(
            count=round(count, 1),
            area_m2=round(area, 1),
            density=round(count / area, 2) if area > 1e-6 else 0.0,
            zones_amber=sum(1 for z in zones if z.risk == "amber"),
            zones_red=sum(1 for z in zones if z.risk == "red"),
        )

    def _debug(self, density: np.ndarray, scale: tuple[float, float]) -> Optional[dict]:
        """Ground truth, when the source is a simulator that knows it.

        Only the synthetic source provides this. It is how the counting stage
        gets scored against a known answer instead of being eyeballed, live,
        while the demo is running.
        """
        truth = self.source.truth_points_px() if self.source else None
        if truth is None:
            return None

        h, w = density.shape
        truth_total = 0
        if truth.size:
            pts = np.asarray(truth, dtype=np.float64) * np.array(scale)
            inside = (
                (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            )
            truth_total = int(inside.sum())

        est = float(density.sum())
        return {
            "truth_count": truth_total,
            "estimated_count": round(est, 1),
            "count_error_pct": round(100 * (est - truth_total) / truth_total, 1) if truth_total else None,
        }

    def _publish_idle(self) -> None:
        """Keep the connection state honest when no frame is arriving."""
        state = self.live.state
        if state is None or self.source is None:
            return
        state.source = self.source.status
        state.latency_ms = int((time.time() - state.ts.timestamp()) * 1000)
        self.live.publish(state)

    def _maybe_purge(self) -> None:
        now = time.time()
        if now - self._last_purge < 600:
            return
        self._last_purge = now
        try:
            removed = self.store.purge(
                settings.ALERT_RETENTION_DAYS,
                settings.AUDIT_RETENTION_DAYS,
                settings.FRAMES_DIR,
                settings.EVIDENCE_RETENTION_HOURS,
            )
            if any(removed.values()):
                log.info("retention purge: %s", removed)
        except Exception:
            log.exception("retention purge failed")
