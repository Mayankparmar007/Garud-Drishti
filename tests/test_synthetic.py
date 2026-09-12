"""Simulator invariants.

The ghat is the spine of the demo, and it is easy to break in a way that still
"runs": if the routing deadlocks, the scene stays permanently jammed and every
downstream number is quietly nonsense while the screen keeps updating. These
tests pin the properties the demo actually depends on.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from uksi.worker.synthetic import (
    MAIN_GATE_X,
    SCENARIO,
    SCENE_H_M,
    WALL_Y,
    WEST_GATE_X,
    SyntheticCrowd,
    phase_at,
)


@lru_cache(maxsize=None)
def run(seconds: float, dt: float = 0.25) -> SyntheticCrowd:
    """Simulate up to ``seconds`` and hand back the crowd.

    Cached: the run is deterministic, and several tests want the same instant.
    The object is shared, so tests must only read from it.
    """
    crowd = SyntheticCrowd()
    for _ in range(int(seconds / dt)):
        crowd.step(dt)
    return crowd


def peak_population(upto: float = 240.0, every: float = 5.0) -> int:
    @lru_cache(maxsize=1)
    def _peak() -> int:
        return max(len(run(t).pos) for t in np.arange(0.0, upto, every))

    return _peak()


def test_arc_ends_recovered() -> None:
    """The last phase must be the quiet one, or the demo has no closing beat."""
    assert SCENARIO[-1].label == "recovered"


def test_inflow_stops_so_the_scene_can_drain() -> None:
    """A recovery that keeps admitting people is not a recovery."""
    assert SCENARIO[-1].inflow_per_s == 0.0
    assert SCENARIO[-2].inflow_per_s == 0.0


def test_crowd_builds_then_clears() -> None:
    """The whole arc: people arrive, pack, and then actually leave."""
    peak = peak_population()
    late = len(run(900).pos)

    assert peak > 300, f"crowd never built up (peak {peak})"
    assert late < peak * 0.5, f"crowd did not clear (peak {peak}, at 900s {late})"


def test_every_agent_is_routed_to_a_real_gate() -> None:
    crowd = run(180)
    assert len(crowd.pos) > 0
    at_gate = np.isclose(crowd.gate_x, MAIN_GATE_X) | np.isclose(crowd.gate_x, WEST_GATE_X)
    assert at_gate.all(), "some agents aim at neither gate"


def test_opening_the_west_corridor_moves_people_west() -> None:
    """Before the corridor opens almost nobody uses it; after, a real share do.

    This is the failure that motivated the routing change: gates were settled at
    spawn, so opening the west gate only ever rerouted people who had not been
    created yet and the crowd already queuing never moved.
    """
    early = run(120)
    west_early = float(np.mean(np.isclose(early.gate_x, WEST_GATE_X)))

    late = run(300)
    west_late = float(np.mean(np.isclose(late.gate_x, WEST_GATE_X)))

    assert west_early < 0.2, f"west gate busy before it opened ({west_early:.2f})"
    assert west_late > 0.4, f"west gate never picked up traffic ({west_late:.2f})"


def test_crowd_is_not_permanently_jammed() -> None:
    """People far from the bottleneck must be able to walk.

    A deadlock pins everyone at the gate at near-zero speed forever. Once the
    scene has drained, whoever is left has room and should be moving.
    """
    crowd = run(900)
    if len(crowd.pos) < 5:
        pytest.skip("scene drained completely")
    speed = np.linalg.norm(crowd.vel, axis=1)
    assert float(speed.mean()) > 0.2, f"crowd is stuck (mean speed {speed.mean():.2f} m/s)"


def test_nobody_is_stranded_outside_the_scene() -> None:
    crowd = run(240)
    y = crowd.pos[:, 1]
    assert y.min() > -1.0, "an agent is stranded above the scene"
    assert y.max() < SCENE_H_M + 1.0, "an agent is stranded below the scene"


def test_gate_geometry_is_inside_the_wall() -> None:
    """Both gaps must lie on the barrier line, not off to one side of it."""
    assert 0 < WEST_GATE_X < MAIN_GATE_X < 30.0
    assert 0 < WALL_Y < SCENE_H_M


def test_phase_lookup_covers_all_time() -> None:
    for t in (0, 39, 40, 99, 100, 199, 200, 264, 265, 329, 330, 1e6):
        assert phase_at(t) is not None
