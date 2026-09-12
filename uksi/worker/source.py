"""Frame sources.

Three kinds behind one interface: an RTSP stream, a video file on loop, and
the synthetic ghat. The worker cannot tell them apart, which is the point --
swapping a clip for a real drone downlink is a config change, not a code path.

Every source runs a reader thread that keeps only the newest frame. A live
stream that outruns the processing budget must drop frames, not queue them;
an operator needs to know what is happening now, and a backlog silently turns
a 1-second display into a 30-second one.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from ..contract import SourceConfig, SourceStatus
from .synthetic import SyntheticCrowd

Frame = np.ndarray


class FrameSource(ABC):
    """Newest-frame-wins reader."""

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._frame_ts: float = 0.0
        self._seq = 0
        self._last_served = -1
        self._connected = False
        self._detail = "starting"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FrameSource":
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _publish(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame
            self._frame_ts = time.time()
            self._seq += 1

    def read(self, timeout: float = 0.0) -> tuple[Optional[Frame], float, bool]:
        """Return (frame, capture_time, is_new).

        ``is_new`` is False when the caller has already processed this frame,
        so the pipeline can idle instead of re-running the model on it.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._frame is not None:
                    fresh = self._seq != self._last_served
                    self._last_served = self._seq
                    return self._frame, self._frame_ts, fresh
            if time.time() >= deadline:
                return None, 0.0, False
            time.sleep(0.01)

    @property
    def status(self) -> SourceStatus:
        return SourceStatus(
            kind=self.cfg.kind,
            uri=self.cfg.uri,
            connected=self._connected,
            detail=self._detail,
        )

    def truth_points_px(self) -> Optional[np.ndarray]:
        """Ground-truth person locations, when the source happens to know them."""
        return None

    @abstractmethod
    def _run(self) -> None: ...


class VideoCaptureSource(FrameSource):
    """OpenCV capture over RTSP or a local file.

    A file is looped and paced to its native frame rate so it behaves like a
    live feed -- otherwise the clip races through at disk speed and every
    speed measurement is wrong by the ratio of the two rates.
    """

    def __init__(self, cfg: SourceConfig):
        super().__init__(cfg)
        self.loop = cfg.kind == "file"

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.cfg.uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._connected = False
                self._detail = f"cannot open {self.cfg.uri}; retrying in {backoff:.0f}s"
                self._stop.wait(backoff)
                backoff = min(backoff * 1.7, 15.0)
                continue

            self._connected = True
            self._detail = "streaming"
            backoff = 1.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            interval = 1.0 / fps if 1.0 < fps < 121.0 else 1.0 / 25.0
            next_due = time.time()

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                    else:
                        self._detail = "stream dropped; reconnecting"
                        break
                self._publish(frame)

                if self.loop:
                    next_due += interval
                    sleep = next_due - time.time()
                    if sleep > 0:
                        self._stop.wait(sleep)
                    else:
                        next_due = time.time()  # we fell behind; do not spiral

            cap.release()
            self._connected = False


class SyntheticSource(FrameSource):
    """The simulated ghat, stepped in real time."""

    def __init__(self, cfg: SourceConfig, fps: float = 6.0, seed: int = 7):
        super().__init__(cfg)
        self.fps = fps
        self.seed = seed
        self.crowd = SyntheticCrowd(seed=seed)
        self._crowd_lock = threading.Lock()

    def _run(self) -> None:
        self._connected = True
        dt = 1.0 / self.fps
        next_due = time.time()
        while not self._stop.is_set():
            with self._crowd_lock:
                self.crowd.step(dt)
                frame = self.crowd.render()
                label, clock = self.crowd.phase_label, self.crowd.t
            self._detail = f"synthetic: {label} (t={clock:.0f}s)"
            self._publish(frame)
            next_due += dt
            sleep = next_due - time.time()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_due = time.time()
        self._connected = False

    def truth_points_px(self) -> Optional[np.ndarray]:
        return self.crowd.truth_points_px()

    def warm_to(self, t_s: float, step: float = 0.25) -> float:
        """Fast-forward the simulation to scenario time ``t_s``.

        Stepping rather than setting the clock, because the two are not the
        same thing: the scenario controls the *inflow rate*, so jumping the
        clock to the saturating phase over an empty ghat would show a gate
        that is saturating with nobody at it. Stepping costs a few seconds of
        wall time and produces a crowd that actually belongs to that phase.

        Rendering is skipped, which is where the real saving is. Going
        backwards restarts from a fresh crowd, since the simulation has no
        history to rewind through.

        This exists for the demo. It is not an operator control, and the UI
        labels it as such.
        """
        t_s = max(0.0, float(t_s))
        with self._crowd_lock:
            if t_s < self.crowd.t:
                self.crowd = SyntheticCrowd(seed=self.seed)
            while self.crowd.t < t_s - 1e-9:
                self.crowd.step(min(step, t_s - self.crowd.t))
            return self.crowd.t


def make_source(cfg: SourceConfig) -> FrameSource:
    if cfg.kind == "synthetic":
        return SyntheticSource(cfg).start()
    return VideoCaptureSource(cfg).start()
