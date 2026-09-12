"""Are agents being routed to the west gate, and if so why don't they get there?"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uksi.worker.synthetic import (  # noqa: E402
    SyntheticCrowd, MAIN_GATE_X, WEST_GATE_X, WALL_Y, SCENE_H_M,
)

crowd = SyntheticCrowd()
for _ in range(int(1200 / 0.25)):
    crowd.step(0.25)

gx = crowd.gate_x
up = crowd.upstream
print(f"people: {len(crowd.pos)}  upstream: {up.sum()}  downstream: {(~up).sum()}")
print(f"gate_x histogram (all):")
h, e = np.histogram(gx, bins=12, range=(0, 30))
for c, l in zip(h, e):
    print(f"  x={l:5.1f}  {'#' * c} {c}")

print(f"\nassigned to west gate (<10): {(gx < 10).sum()}")
print(f"assigned to main gate (>=10): {(gx >= 10).sum()}")
print(f"\nof those assigned west, upstream (returning): {((gx < 10) & up).sum()}")
print(f"of those assigned west, downstream          : {((gx < 10) & ~up).sum()}")

# For agents assigned the west gate, what are they actually doing?
w = gx < 10
if w.any():
    p = crowd.pos[w]
    v = crowd.vel[w]
    print(f"\nwest-assigned agents: {w.sum()}")
    print(f"  x: {p[:,0].min():.2f} .. {p[:,0].max():.2f}")
    print(f"  y: {p[:,1].min():.2f} .. {p[:,1].max():.2f}")
    print(f"  speed mean: {np.linalg.norm(v,axis=1).mean():.3f}")
    print(f"  downstream ones heading to y={WALL_Y+1.0}, currently y mean "
          f"{p[~up[w],1].mean() if (~up[w]).any() else float('nan'):.2f}")
else:
    print("\nno agents assigned to the west gate at all")
