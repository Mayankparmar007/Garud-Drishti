"""Where does the speed go? Diagnostic for the simulator's low-speed tail."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uksi.worker.synthetic import SyntheticCrowd, FREE_SPEED, JAM_DENSITY  # noqa: E402

CROWD = SyntheticCrowd()
T_S = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
for _ in range(int(T_S / 0.25)):
    CROWD.step(0.25)

n = len(CROWD.pos)
if n == 0:
    print(f"t={T_S:.0f}s: no people left")
    raise SystemExit(0)
speed = np.linalg.norm(CROWD.vel, axis=1)
dens = CROWD.local_density
print(f"t={T_S:.0f}s  people: {n}")
print(f"speed  mean   : {speed.mean():.3f}  median {np.median(speed):.3f}  "
      f"p90 {np.percentile(speed, 90):.3f}  max {speed.max():.3f}")
print(f"v_max  mean   : {(FREE_SPEED * np.clip(1 - dens / JAM_DENSITY, 0, 1)).mean():.3f}")
print(f"local density : mean {dens.mean():.3f}  p50 {np.median(dens):.3f}  "
      f"p95 {np.percentile(dens, 95):.3f}  max {dens.max():.3f}")
print(f"at jam (v=0)  : {(dens >= JAM_DENSITY).sum()} people")

y = CROWD.pos[:, 1]
print(f"y position    : min {y.min():.2f}  max {y.max():.2f}")
print(f"y > 16.5 (stuck at exit): {(y > 16.5).sum()}")

from uksi.worker.synthetic import SCENE_H_M, SCENE_W_M, WALL_Y  # noqa: E402
print(f"scene        : {SCENE_W_M} x {SCENE_H_M} m, wall at y={WALL_Y}")
