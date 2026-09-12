"""Print the simulator's population arc. Dev tool, not part of the demo."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uksi.worker.synthetic import SyntheticCrowd, phase_at  # noqa: E402

crowd = SyntheticCrowd()
marks = [0, 20, 40, 60, 80, 100, 130, 160, 200, 230, 265, 300, 340, 400, 500, 700, 900, 1200]
dt, t, mi = 0.25, 0.0, 0
print(f"{'t':>6}  {'phase':<22} {'people':>7}  {'mean p/m2':>9}  {'mean m/s':>8}")
while mi < len(marks):
    if t >= marks[mi] - 1e-9:
        n = len(crowd.pos)
        area = 30.0 * 17.0
        spd = float(np.sqrt((crowd.vel ** 2).sum(axis=1)).mean()) if n else 0.0
        print(f"{t:>6.0f}  {crowd.phase_label:<22} {n:>7}  {n / area:>9.2f}  {spd:>8.2f}")
        mi += 1
        continue
    crowd.step(dt)
    t += dt
