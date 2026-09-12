"""The hand-off between the worker thread and the API.

The worker runs at the sample rate and the dashboard is pushed at 1 Hz. Those
two rates are deliberately decoupled here: a slow client must never slow the
pipeline down, and a fast pipeline must never flood a client.

The rolling display frame lives in memory and is never written to disk. The
only imagery that reaches storage is a downscaled evidence thumbnail attached
to a fired alert.
"""

from __future__ import annotations

import threading
from typing import Optional

from .contract import StatePayload


class LiveState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[StatePayload] = None
        self._wire: Optional[dict] = None
        self._frame_jpeg: bytes = b""
        self._frame_seq = 0
        self._heat_png: bytes = b""
        self._heat_seq = 0

    # -- state ------------------------------------------------------------

    def publish(self, state: StatePayload) -> None:
        wire = state.to_wire()  # serialise once, on the worker's time, not per client
        with self._lock:
            self._state, self._wire = state, wire

    @property
    def state(self) -> Optional[StatePayload]:
        with self._lock:
            return self._state

    @property
    def wire(self) -> Optional[dict]:
        with self._lock:
            return self._wire

    # -- imagery ----------------------------------------------------------

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame_jpeg = jpeg
            self._frame_seq += 1

    @property
    def frame(self) -> tuple[bytes, int]:
        with self._lock:
            return self._frame_jpeg, self._frame_seq

    def set_heat(self, png: bytes) -> None:
        with self._lock:
            self._heat_png = png
            self._heat_seq += 1

    @property
    def heat(self) -> tuple[bytes, int]:
        with self._lock:
            return self._heat_png, self._heat_seq
