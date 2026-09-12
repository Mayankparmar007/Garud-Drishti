"""Paths and runtime tunables.

Everything the system writes lives under ``data/runtime`` so a single delete
resets the deployment. Nothing here is scene-specific -- scene config lives in
SQLite and is edited through the setup UI.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("UKSI_DATA_DIR", ROOT / "data"))
CLIPS_DIR = DATA_DIR / "clips"
WEIGHTS_DIR = DATA_DIR / "weights"
RUNTIME_DIR = DATA_DIR / "runtime"
FRAMES_DIR = RUNTIME_DIR / "frames"
DB_PATH = RUNTIME_DIR / "uksi.db"

FRONTEND_DIR = ROOT / "frontend"
CONFIG_DIR = ROOT / "config"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Frames pulled off the source per second. The drone downlink is 25-30 fps;
# crowd density does not change meaningfully between adjacent frames, so we
# sample down hard and spend the budget on the model instead.
SAMPLE_FPS = float(os.getenv("UKSI_SAMPLE_FPS", "3"))

# Longest edge the density model sees. Dropping this is the first lever if
# latency climbs.
INFERENCE_LONG_EDGE = int(os.getenv("UKSI_INFER_LONG_EDGE", "960"))

# Rate the dashboard is pushed state at.
PUBLISH_HZ = float(os.getenv("UKSI_PUBLISH_HZ", "1"))

# Density backend: auto | csrnet | yolo | blob
DENSITY_BACKEND = os.getenv("UKSI_DENSITY_BACKEND", "auto")

CSRNET_WEIGHTS = Path(os.getenv("UKSI_CSRNET_WEIGHTS", WEIGHTS_DIR / "csrnet.pth"))
YOLO_WEIGHTS = os.getenv("UKSI_YOLO_WEIGHTS", str(WEIGHTS_DIR / "yolo11n.pt"))

# Default source used when no scene has been configured yet.
DEFAULT_SOURCE_KIND = os.getenv("UKSI_SOURCE_KIND", "synthetic")
DEFAULT_SOURCE_URI = os.getenv("UKSI_SOURCE_URI", "rtsp://127.0.0.1:8554/live")

# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

# Raw video is never written to disk. The rolling operator display frame lives
# in memory only. The one derived artefact that reaches storage is a downscaled
# evidence thumbnail attached to a fired alert, bounded by the retention below
# and deliberately too coarse to identify a face.
THUMB_LONG_EDGE = int(os.getenv("UKSI_THUMB_LONG_EDGE", "480"))
EVIDENCE_RETENTION_HOURS = float(os.getenv("UKSI_EVIDENCE_RETENTION_HOURS", "24"))
ALERT_RETENTION_DAYS = float(os.getenv("UKSI_ALERT_RETENTION_DAYS", "30"))
AUDIT_RETENTION_DAYS = float(os.getenv("UKSI_AUDIT_RETENTION_DAYS", "90"))

# History kept in memory per zone, in samples. 600 @ 3fps = ~3 min of trend.
HISTORY_SAMPLES = int(os.getenv("UKSI_HISTORY_SAMPLES", "600"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, CLIPS_DIR, WEIGHTS_DIR, RUNTIME_DIR, FRAMES_DIR):
        d.mkdir(parents=True, exist_ok=True)
