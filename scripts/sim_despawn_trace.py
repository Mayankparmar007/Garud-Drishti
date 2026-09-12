"""Watch gate_x as agents are rerouted on despawn."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uksi.worker.synthetic as S  # noqa: E402

print("west_share in 'clearing' phase:", S.phase_at(300).west_share)
print("west_share in 'recovered' phase:", S.phase_at(2000).west_share)
print()

# Patch _despawn to report what it does.
orig = S.SyntheticCrowd._despawn


def traced(self, phase):
    before = len(self.pos)
    y = self.pos[:, 1]
    gone = (y > S.SCENE_H_M + 0.5) | (y < -0.5)
    if gone.any():
        returning = gone & (y > S.SCENE_H_M) & (self.rng.random(len(self.pos)) < 0.22)
        n_ret = int(returning.sum())
        print(f"  t={self.t:7.1f} gone={int(gone.sum()):3d} returning={n_ret:3d} "
              f"west_share={phase.west_share:.2f} -> west assigned="
              f"{int((self.rng.random(n_ret) < phase.west_share).sum()) if n_ret else 0}")
    return orig(self, phase)


S.SyntheticCrowd._despawn = traced

crowd = S.SyntheticCrowd()
for i in range(int(400 / 0.25)):
    crowd.step(0.25)

gx = crowd.gate_x
print(f"\nafter 400s: people={len(crowd.pos)}  west-assigned={(gx < 10).sum()}")
print(f"gate_x values present: {sorted(set(np.round(gx, 2)))[:12]}")
