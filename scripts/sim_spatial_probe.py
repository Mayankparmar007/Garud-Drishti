"""Spatial distribution of the crowd once it should have drained."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uksi.worker.synthetic import (  # noqa: E402
    SyntheticCrowd, SCENE_W_M, SCENE_H_M, WALL_Y, MAIN_GATE_X, WEST_GATE_X,
)

crowd = SyntheticCrowd()
for _ in range(int(1200 / 0.25)):
    crowd.step(0.25)

pos = crowd.pos
print(f"people: {len(pos)}")
print(f"x range: {pos[:,0].min():.2f} .. {pos[:,0].max():.2f}  (scene 0..{SCENE_W_M})")
print(f"y range: {pos[:,1].min():.2f} .. {pos[:,1].max():.2f}  (wall {WALL_Y}, h {SCENE_H_M})")

hx, ex = np.histogram(pos[:, 0], bins=15, range=(0, SCENE_W_M))
print("\nx histogram:")
for c, l in zip(hx, ex):
    print(f"  x={l:5.1f}  {'#' * c} {c}")

hy, ey = np.histogram(pos[:, 1], bins=12, range=(0, SCENE_H_M))
print("\ny histogram:")
for c, l in zip(hy, ey):
    print(f"  y={l:5.1f}  {'#' * c} {c}")

near_main = np.abs(pos[:, 0] - MAIN_GATE_X) < 1.5
near_west = np.abs(pos[:, 0] - WEST_GATE_X) < 1.25
print(f"\nnear main gate x={MAIN_GATE_X}: {near_main.sum()}")
print(f"near west gate x={WEST_GATE_X}: {near_west.sum()}")
print(f"upstream (returning): {crowd.upstream.sum()}")
print(f"downstream          : {(~crowd.upstream).sum()}")

# are they actually moving toward a target?
tgt = crowd._targets()
want = tgt - pos
print(f"\ntarget y: min {tgt[:,1].min():.2f} max {tgt[:,1].max():.2f}")
print(f"dist to target: mean {np.linalg.norm(want,axis=1).mean():.2f} m")
