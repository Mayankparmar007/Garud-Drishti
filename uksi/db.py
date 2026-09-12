"""SQLite storage: alerts, audit log, scene config.

Three tables and nothing else. No frames, no tracks, no per-person rows -- the
schema is itself part of the privacy position, because a database that has no
column for identity cannot accumulate one by accident.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .contract import Alert, SceneConfig, iso, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_config (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    json       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id          TEXT PRIMARY KEY,
    seq         INTEGER NOT NULL,
    zone        TEXT NOT NULL,
    zone_name   TEXT NOT NULL,
    severity    TEXT NOT NULL,
    metric      TEXT NOT NULL,
    message     TEXT NOT NULL,
    action      TEXT NOT NULL DEFAULT '',
    thumb       TEXT,
    ts          TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    state       TEXT NOT NULL,
    escalations INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS alerts_ts ON alerts (ts DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT NOT NULL,
    actor  TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS audit_ts ON audit_log (ts DESC);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- scene config -----------------------------------------------------

    def load_scene(self) -> Optional[SceneConfig]:
        with self._lock:
            row = self._conn.execute("SELECT json FROM scene_config WHERE id = 1").fetchone()
        if row is None:
            return None
        try:
            return SceneConfig.model_validate_json(row["json"])
        except Exception:
            return None  # a config we can no longer parse is not worth crashing over

    def save_scene(self, scene: SceneConfig) -> None:
        payload = scene.model_dump_json()
        with self._lock:
            self._conn.execute(
                "INSERT INTO scene_config (id, json, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET json = excluded.json, updated_at = excluded.updated_at",
                (payload, iso(utcnow())),
            )
            self._conn.commit()

    # -- alerts -----------------------------------------------------------

    def upsert_alert(self, alert: Alert) -> None:
        seq = int(alert.id[1:]) if alert.id[1:].isdigit() else 0
        with self._lock:
            self._conn.execute(
                """INSERT INTO alerts
                   (id, seq, zone, zone_name, severity, metric, message, action, thumb,
                    ts, first_seen, state, escalations, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     severity=excluded.severity, metric=excluded.metric,
                     message=excluded.message, action=excluded.action,
                     thumb=excluded.thumb, ts=excluded.ts, state=excluded.state,
                     escalations=excluded.escalations, resolved_at=excluded.resolved_at""",
                (
                    alert.id, seq, alert.zone, alert.zone_name, alert.severity, alert.metric,
                    alert.message, alert.action, alert.thumb, iso(alert.ts), iso(alert.first_seen),
                    alert.state, alert.escalations,
                    iso(alert.resolved_at) if alert.resolved_at else None,
                ),
            )
            self._conn.commit()

    def max_alert_seq(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT MAX(seq) AS m FROM alerts").fetchone()
        return int(row["m"] or 0)

    def recent_alerts(self, limit: int = 50) -> list[Alert]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            Alert(
                id=r["id"], zone=r["zone"], zone_name=r["zone_name"], severity=r["severity"],
                metric=r["metric"], message=r["message"], action=r["action"], thumb=r["thumb"],
                ts=r["ts"], first_seen=r["first_seen"], state=r["state"],
                escalations=r["escalations"], resolved_at=r["resolved_at"],
            )
            for r in rows
        ]

    # -- audit ------------------------------------------------------------

    def audit(self, actor: str, action: str, detail: str = "") -> None:
        """Record an access or a change.

        Deliberately not called on every websocket frame: a log nobody can read
        is not accountability. What is recorded is session open and close,
        every scene-config change, and every retrieval of a stored evidence
        image -- the things that would matter in an enquiry.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (ts, actor, action, detail) VALUES (?,?,?,?)",
                (iso(utcnow()), actor, action, detail),
            )
            self._conn.commit()

    def audit_tail(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- retention --------------------------------------------------------

    def purge(self, alert_days: float, audit_days: float, frames_dir: Path,
              evidence_hours: float) -> dict[str, int]:
        """Enforce retention. Called on a timer, not left to good intentions."""
        now = utcnow()
        alert_cut = iso(now - timedelta(days=alert_days))
        audit_cut = iso(now - timedelta(days=audit_days))

        with self._lock:
            a = self._conn.execute("DELETE FROM alerts WHERE ts < ?", (alert_cut,)).rowcount
            b = self._conn.execute("DELETE FROM audit_log WHERE ts < ?", (audit_cut,)).rowcount
            self._conn.commit()

        cutoff = (now - timedelta(hours=evidence_hours)).timestamp()
        removed = 0
        if frames_dir.exists():
            for f in frames_dir.glob("*.jpg"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        return {"alerts": a, "audit": b, "evidence": removed}
