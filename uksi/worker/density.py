"""Density estimation.

Density regression rather than object detection, for two reasons. At drone
altitude a head is a handful of pixels and merges with its neighbours in
exactly the conditions we care about, so detectors fall apart precisely when
the answer matters. And a density surface carries no identity: there is no box
around a person, no track, nothing to re-identify. The privacy argument and
the accuracy argument point the same way.

Three interchangeable backends, so a weights problem is a config change rather
than a rewrite:

  csrnet  pretrained crowd-counting network, the real one
  yolo    person detection splatted into a density surface, the fallback
  blob    classical connected components, for the synthetic scene and for
          running the whole pipeline with nothing downloaded

Every backend returns a float32 map the size of the frame whose sum is the
estimated number of people in it. Downstream code only knows that contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_preserving_sum(dmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a density map without changing how many people it says there are.

    Interpolation preserves per-pixel values, not totals, so the ratio of
    areas has to be put back by hand. Skipping this is a silent 64x count
    error with a network that outputs at 1/8 scale.
    """
    src_total = float(dmap.sum())
    out = cv2.resize(dmap, (width, height), interpolation=cv2.INTER_LINEAR)
    out_total = float(out.sum())
    if out_total > 1e-9:
        out *= src_total / out_total
    return out.astype(np.float32)


class DensityBackend(ABC):
    name = "abstract"
    detail = ""

    @abstractmethod
    def density(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel density map, same H x W as the frame, summing to a count."""


# ---------------------------------------------------------------------------
# CSRNet
# ---------------------------------------------------------------------------


def _build_csrnet():
    import torch.nn as nn

    def layers(spec, in_channels, dilation=1):
        mods = []
        for v in spec:
            if v == "M":
                mods.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                mods.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation))
                mods.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*mods)

    class CSRNet(nn.Module):
        """VGG16 frontend to 1/8 scale, dilated backend, 1x1 regression head."""

        def __init__(self):
            super().__init__()
            self.frontend = layers(
                [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512], 3
            )
            self.backend = layers([512, 512, 512, 256, 128, 64], 512, dilation=2)
            self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x):
            return self.output_layer(self.backend(self.frontend(x)))

    return CSRNet()


def _clean_state_dict(raw: dict) -> dict:
    """Published checkpoints disagree about wrappers. Normalise the keys."""
    for key in ("state_dict", "model", "net", "model_state_dict"):
        if key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    out = {}
    for k, v in raw.items():
        for prefix in ("module.", "model.", "net."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        out[k] = v
    return out


class CSRNetBackend(DensityBackend):
    name = "csrnet"

    def __init__(self, weights: Path, count_scale: float = 1.0, threads: int = 4):
        import torch

        self.torch = torch
        torch.set_num_threads(max(1, threads))
        self.count_scale = count_scale

        self.model = _build_csrnet()
        state = _clean_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        loaded = len(self.model.state_dict()) - len(missing)
        if loaded == 0:
            raise RuntimeError(f"no matching tensors in {weights}")
        if missing or unexpected:
            log.warning(
                "csrnet: %d/%d tensors loaded (%d missing, %d unexpected)",
                loaded, len(self.model.state_dict()), len(missing), len(unexpected),
            )
        self.model.eval()
        self.detail = f"{weights.name}, {loaded} tensors"

    def density(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        with self.torch.inference_mode():
            out = self.model(tensor)
        dmap = out[0, 0].cpu().numpy().astype(np.float32)
        dmap = np.clip(dmap, 0.0, None) * self.count_scale
        return resize_preserving_sum(dmap, w, h)


# ---------------------------------------------------------------------------
# YOLO person detection
# ---------------------------------------------------------------------------


class YoloBackend(DensityBackend):
    """Fallback. Detections are splatted into a density surface, and the boxes
    are discarded immediately -- nothing per-person leaves this function."""

    name = "yolo"

    def __init__(self, weights: str, count_scale: float = 1.0, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.count_scale = count_scale
        self.detail = f"{Path(weights).name}, conf {conf}"

    def density(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        dmap = np.zeros((h, w), dtype=np.float32)
        res = self.model.predict(frame_bgr, conf=self.conf, classes=[0], verbose=False)
        if not res:
            return dmap

        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return dmap
        xyxy = boxes.xyxy.cpu().numpy()

        for x1, y1, x2, y2 in xyxy:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            sigma = max(2.0, min(x2 - x1, y2 - y1) / 4.0)
            r = int(sigma * 3)
            xs = np.arange(max(0, int(cx) - r), min(w, int(cx) + r + 1))
            ys = np.arange(max(0, int(cy) - r), min(h, int(cy) + r + 1))
            if xs.size == 0 or ys.size == 0:
                continue
            gx = np.exp(-((xs - cx) ** 2) / (2 * sigma**2))
            gy = np.exp(-((ys - cy) ** 2) / (2 * sigma**2))
            patch = np.outer(gy, gx).astype(np.float32)
            total = patch.sum()
            if total > 1e-9:
                dmap[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] += patch / total
        return dmap * self.count_scale


# ---------------------------------------------------------------------------
# Classical blobs
# ---------------------------------------------------------------------------


class BlobBackend(DensityBackend):
    """Connected components, sized against a known per-person footprint.

    Dev backend: it needs people darker than their background, which is true
    of the synthetic ghat and of very little real footage. It is here so the
    entire pipeline runs with nothing downloaded, and so the density stage can
    be scored against simulator ground truth.

    Merged blobs are handled by area rather than by counting components, which
    is the one idea it shares with real density regression: at high density you
    cannot find individuals, so stop trying and measure the mass.
    """

    name = "blob"

    def __init__(self, px_per_person: float = 78.0, offset: float = 40.0, count_scale: float = 1.0):
        self.px_per_person = px_per_person
        self.offset = offset
        self.count_scale = count_scale
        self.detail = f"{px_per_person:.0f} px/person (dev backend)"

    def density(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        thresh = float(np.median(gray)) - self.offset
        mask = (gray < thresh).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dmap = np.zeros(gray.shape, dtype=np.float32)
        if n_lab <= 1:
            return dmap

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        # Spread each component's estimated headcount evenly over its pixels:
        # correct in aggregate, and never claims to know where one person ends.
        counts = np.maximum(1.0, areas / self.px_per_person)
        per_pixel = np.zeros(n_lab, dtype=np.float32)
        per_pixel[1:] = counts / np.maximum(areas, 1.0)
        dmap = per_pixel[labels]
        return dmap * self.count_scale


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def make_backend(
    prefer: str = "auto",
    csrnet_weights: Optional[Path] = None,
    yolo_weights: str = "",
    count_scale: float = 1.0,
) -> DensityBackend:
    """Pick a backend, degrading rather than failing.

    Order is deliberate: the real model first, then the detector fallback, then
    the classical one that always works. A loud log line records which we got,
    and the choice is surfaced in the UI -- an operator should never have to
    guess whether they are looking at model output or a stand-in.
    """
    order = [prefer] if prefer != "auto" else ["csrnet", "yolo", "blob"]
    errors = []

    for kind in order:
        try:
            if kind == "csrnet":
                if csrnet_weights is None or not Path(csrnet_weights).exists():
                    raise FileNotFoundError(f"no weights at {csrnet_weights}")
                be = CSRNetBackend(Path(csrnet_weights), count_scale=count_scale)
            elif kind == "yolo":
                be = YoloBackend(yolo_weights, count_scale=count_scale)
            elif kind == "blob":
                be = BlobBackend(count_scale=count_scale)
            else:
                raise ValueError(f"unknown density backend {kind!r}")
            log.info("density backend: %s (%s)", be.name, be.detail)
            return be
        except Exception as exc:
            errors.append(f"{kind}: {exc}")
            log.warning("density backend %s unavailable: %s", kind, exc)

    raise RuntimeError("no density backend available -- " + "; ".join(errors))
