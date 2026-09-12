"""A synthetic ghat for development and stress-testing.

Not a toy loop of coloured dots: agents steer toward a gate, repel each other,
and slow down when packed, so a queue forms at a bottleneck on its own. That
matters because the thing we most need to exercise -- density rising while
speed collapses -- is exactly the thing no public clip reliably contains.

It also gives us ground truth. The simulator knows where every agent is, so
the density stage can be scored against a known answer rather than eyeballed.

The scene is viewed straight down from a hover, so pixels map to metres by a
constant scale. That makes the calibration exact and lets an area error in the
pipeline show up as an obvious number rather than a plausible one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # cKDTree makes neighbour lookup cheap; fall back to brute force.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    cKDTree = None

# --- Scene geometry, in metres -------------------------------------------
SCENE_W_M = 30.0
SCENE_H_M = 17.0
PX_PER_M = 32.0
FRAME_W = int(SCENE_W_M * PX_PER_M)  # 960
FRAME_H = int(SCENE_H_M * PX_PER_M)  # 544

WALL_Y = 9.5  # barrier line across the ghat
WALL_THICK = 0.35

MAIN_GATE_X = 21.0
MAIN_GATE_W = 3.0
WEST_GATE_X = 5.0
WEST_GATE_W = 2.5

FREE_SPEED = 1.25  # m/s unimpeded walking
PERSON_R = 0.22  # m, body radius used for repulsion
JAM_DENSITY = 6.0  # p/m2 at which movement stops entirely
SENSE_R = 1.0  # m, radius the local density is measured over


@dataclass
class Phase:
    """One beat of the demo arc."""

    until_s: float
    inflow_per_s: float
    west_share: float  # fraction routed to the west corridor
    label: str


# The arc the demo is built backwards from: calm, a procession arrives, the
# gate cannot absorb it, an operator opens the west corridor, it clears.
#
# Inflow has to fall to zero for that last word to be true. Held at a trickle
# the scene reaches a jammed steady state and stays there -- the crowd packs
# until bodies cannot overlap, mean speed parks under the breakdown threshold,
# and the zone never goes green. The recovery is a beat in the demo, not a
# caption, so the script stops adding people and lets the ghat drain.
SCENARIO: tuple[Phase, ...] = (
    Phase(40, 1.2, 0.10, "calm"),
    Phase(100, 3.8, 0.10, "procession arriving"),
    Phase(200, 5.2, 0.08, "gate saturating"),
    Phase(265, 2.0, 0.70, "west corridor opened"),
    Phase(330, 0.0, 0.60, "clearing"),
    Phase(1e9, 0.0, 0.60, "recovered"),
)


def phase_at(t: float) -> Phase:
    for p in SCENARIO:
        if t < p.until_s:
            return p
    return SCENARIO[-1]


class SyntheticCrowd:
    """Agent-based crowd over a gated ghat."""

    def __init__(self, seed: int = 7, max_agents: int = 1400):
        self.rng = np.random.default_rng(seed)
        self.max_agents = max_agents
        self.t = 0.0
        self.pos = np.zeros((0, 2), dtype=np.float64)
        self.vel = np.zeros((0, 2), dtype=np.float64)
        self.gate_x = np.zeros(0, dtype=np.float64)  # which gap each agent aims at
        self.route_u = np.zeros(0, dtype=np.float64)  # fixed draw deciding west vs main
        self.upstream = np.zeros(0, dtype=bool)  # returning from the river
        self.local_density = np.zeros(0, dtype=np.float64)
        self._spawn_debt = 0.0
        self._bg = self._render_background()

    # -- geometry helpers -------------------------------------------------

    @staticmethod
    def _in_gap(x: np.ndarray) -> np.ndarray:
        main = np.abs(x - MAIN_GATE_X) < MAIN_GATE_W / 2
        west = np.abs(x - WEST_GATE_X) < WEST_GATE_W / 2
        return main | west

    def _targets(self) -> np.ndarray:
        """Where each agent is currently trying to get to."""
        n = len(self.pos)
        tgt = np.empty((n, 2), dtype=np.float64)
        if n == 0:
            return tgt

        y = self.pos[:, 1]
        down = ~self.upstream

        # Heading downstream: aim for your gap, then the exit below.
        before = down & (y < WALL_Y - 0.3)
        tgt[before, 0] = self.gate_x[before]
        tgt[before, 1] = WALL_Y + 1.0
        after = down & ~before
        tgt[after, 0] = self.pos[after, 0]
        tgt[after, 1] = SCENE_H_M + 1.0

        # Heading upstream: back through the gap and up onto the ghat.
        up_before = self.upstream & (y > WALL_Y + 0.3)
        tgt[up_before, 0] = self.gate_x[up_before]
        tgt[up_before, 1] = WALL_Y - 1.0
        up_after = self.upstream & ~up_before
        tgt[up_after, 0] = self.pos[up_after, 0]
        tgt[up_after, 1] = -1.0
        return tgt

    def _route(self, phase: Phase) -> None:
        """Re-aim everyone at a gate, from a fixed draw per person.

        The gate was originally settled when an agent spawned, which meant
        opening the west corridor only ever rerouted people who had not been
        born yet. The crowd already queuing kept funnelling into the main gate
        while people came back the other way through it, and the two flows
        locked solid -- a permanently jammed scene that the script kept
        labelling "recovered".

        Each agent carries a fixed uniform draw, so re-deciding against the
        current phase moves a stable share of them west the moment the corridor
        opens, and does not oscillate them back and forth frame to frame. The
        draw is per person and never redrawn, so who goes west is arbitrary but
        each individual's choice is fixed -- which is what keeps it stable.
        """
        if len(self.pos) == 0:
            return
        west = self.route_u < phase.west_share
        self.gate_x = np.where(west, WEST_GATE_X, MAIN_GATE_X)

    # -- simulation -------------------------------------------------------

    def _spawn(self, dt: float, phase: Phase) -> None:
        self._spawn_debt += phase.inflow_per_s * dt
        n_new = int(self._spawn_debt)
        if n_new <= 0:
            return
        self._spawn_debt -= n_new
        n_new = min(n_new, max(0, self.max_agents - len(self.pos)))
        if n_new == 0:
            return

        x = self.rng.uniform(2.0, SCENE_W_M - 2.0, n_new)
        y = self.rng.uniform(0.1, 0.9, n_new)
        # Who ends up going west is decided by a fixed draw per person and
        # re-read against the current phase every step, so the corridor can be
        # opened onto a crowd that is already standing there.
        u = self.rng.random(n_new)
        self.route_u = np.concatenate([self.route_u, u])

        self.pos = np.vstack([self.pos, np.column_stack([x, y])])
        self.vel = np.vstack([self.vel, np.zeros((n_new, 2))])
        self.gate_x = np.concatenate([self.gate_x, np.full(n_new, MAIN_GATE_X)])
        self.upstream = np.concatenate([self.upstream, np.zeros(n_new, dtype=bool)])

    def _repulsion(self) -> np.ndarray:
        """Short-range personal-space force. This is what makes a queue pack."""
        n = len(self.pos)
        force = np.zeros((n, 2), dtype=np.float64)
        self.local_density = np.zeros(n, dtype=np.float64)
        if n < 2:
            return force

        radius = PERSON_R * 3.2
        if cKDTree is not None:
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(radius, output_type="ndarray")
            # Local density over a 1 m sensing disc, used by the fundamental
            # diagram below. Counting includes self, which is correct: one
            # person alone occupies their own square metre.
            self.local_density = tree.query_ball_point(
                self.pos, SENSE_R, return_length=True
            ) / (np.pi * SENSE_R**2)
        else:  # pragma: no cover
            d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
            self.local_density = (d < SENSE_R).sum(axis=1) / (np.pi * SENSE_R**2)
            iu = np.triu_indices(n, k=1)
            pairs = np.column_stack([iu[0][d[iu] < radius], iu[1][d[iu] < radius]])
        if len(pairs) == 0:
            return force

        a, b = pairs[:, 0], pairs[:, 1]
        delta = self.pos[a] - self.pos[b]
        dist = np.linalg.norm(delta, axis=1)
        dist = np.where(dist < 1e-6, 1e-6, dist)
        strength = np.clip((radius - dist) / radius, 0, 1) ** 2 * 3.0
        push = delta / dist[:, None] * strength[:, None]
        np.add.at(force, a, push)
        np.add.at(force, b, -push)
        return force

    def _wall_steering(self) -> np.ndarray:
        """Funnel agents sideways toward their gap as they near the barrier."""
        n = len(self.pos)
        force = np.zeros((n, 2), dtype=np.float64)
        if n == 0:
            return force
        dy = np.abs(self.pos[:, 1] - WALL_Y)
        near = dy < 3.0
        if not near.any():
            return force
        dx = self.gate_x[near] - self.pos[near, 0]
        urgency = (3.0 - dy[near]) / 3.0
        force[near, 0] = np.clip(dx, -2.5, 2.5) * urgency * 1.4
        return force

    def _apply_wall(self, prev_y: np.ndarray) -> None:
        """Hard constraint: the barrier is solid except at the gaps.

        Without this the crowd walks through the wall and no queue ever forms.
        """
        if len(self.pos) == 0:
            return
        x, y = self.pos[:, 0], self.pos[:, 1]
        blocked = ~self._in_gap(x)
        crossing_down = blocked & (prev_y <= WALL_Y) & (y > WALL_Y - WALL_THICK)
        crossing_up = blocked & (prev_y >= WALL_Y) & (y < WALL_Y + WALL_THICK)
        self.pos[crossing_down, 1] = WALL_Y - WALL_THICK
        self.pos[crossing_up, 1] = WALL_Y + WALL_THICK
        self.vel[crossing_down | crossing_up, 1] *= -0.05

    def _despawn(self, phase: Phase) -> None:
        n = len(self.pos)
        if n == 0:
            return
        y = self.pos[:, 1]
        gone = (y > SCENE_H_M + 0.5) | (y < -0.5)
        if not gone.any():
            return

        # A share of those leaving turn around and come back up: people
        # returning from the river, which is what creates opposing flow.
        returning = gone & (y > SCENE_H_M) & (self.rng.random(n) < 0.22)
        self.pos[returning, 1] = SCENE_H_M - 0.2
        self.upstream[returning] = True

        keep = ~(gone & ~returning)
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.route_u = self.route_u[keep]
        self.gate_x, self.upstream = self.gate_x[keep], self.upstream[keep]

    def _resolve_overlaps(self, iterations: int = 3, min_sep: float = 0.42) -> None:
        """Bodies cannot occupy the same ground.

        This is what actually caps density. Hexagonal packing at 0.42 m
        spacing is about 6.5 p/m2, which is the right order for the worst
        real-world crush measurements -- without it the crowd compresses to
        arbitrary densities and every number downstream is fiction.
        """
        if len(self.pos) < 2 or cKDTree is None:
            return
        for _ in range(iterations):
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(min_sep, output_type="ndarray")
            if len(pairs) == 0:
                return
            a, b = pairs[:, 0], pairs[:, 1]
            delta = self.pos[a] - self.pos[b]
            dist = np.linalg.norm(delta, axis=1)
            coincident = dist < 1e-6
            if coincident.any():  # nudge exact overlaps apart in a random direction
                delta[coincident] = self.rng.normal(0, 1e-3, (int(coincident.sum()), 2))
                dist = np.linalg.norm(delta, axis=1)
            dist = np.where(dist < 1e-9, 1e-9, dist)
            corr = ((min_sep - dist) / 2.0)[:, None] * (delta / dist[:, None])
            np.add.at(self.pos, a, corr)
            np.add.at(self.pos, b, -corr)

    def step(self, dt: float) -> None:
        phase = phase_at(self.t)
        self._spawn(dt, phase)
        if len(self.pos) == 0:
            self.t += dt
            return

        repulse = self._repulsion()  # also refreshes self.local_density

        # Fundamental diagram: walking speed falls as local density rises and
        # reaches zero at jam. Applied to the driving term only -- a packed
        # crowd must still be able to push apart. Capping the net velocity
        # instead freezes the jam permanently and lets density run away.
        v_max = FREE_SPEED * np.clip(1.0 - self.local_density / JAM_DENSITY, 0.0, 1.0)

        tgt = self._targets()
        want = tgt - self.pos
        norm = np.linalg.norm(want, axis=1, keepdims=True)
        want = want / np.where(norm < 1e-6, 1e-6, norm) * v_max[:, None]

        crawl = np.clip(v_max[:, None] / FREE_SPEED, 0.15, 1.0)
        desired = want + repulse + self._wall_steering() * crawl
        desired += self.rng.normal(0, 0.10, desired.shape)  # gait jitter

        self.vel += (desired - self.vel) * min(1.0, dt / 0.45)
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        over = speed > FREE_SPEED
        self.vel = np.where(over, self.vel / np.where(speed < 1e-6, 1e-6, speed) * FREE_SPEED, self.vel)

        prev_y = self.pos[:, 1].copy()
        self.pos += self.vel * dt
        self.pos[:, 0] = np.clip(self.pos[:, 0], 0.2, SCENE_W_M - 0.2)
        self._resolve_overlaps()
        self._apply_wall(prev_y)
        self._despawn(phase)
        self._route(phase)
        self.t += dt

    # -- rendering --------------------------------------------------------

    def _render_background(self) -> np.ndarray:
        bg = np.full((FRAME_H, FRAME_W, 3), 168, dtype=np.uint8)
        # Ghat steps: faint horizontal banding above the barrier.
        for i in range(0, int(WALL_Y * PX_PER_M), 26):
            bg[i : i + 13, :, :] = 158
        # Apron below the barrier, slightly darker stone.
        bg[int((WALL_Y + WALL_THICK) * PX_PER_M) :, :, :] = 148
        # The barrier itself, with the two gaps knocked out. Deliberately
        # lighter than the background: real barricades are pale metal, and it
        # keeps the barrier from reading as a dense crowd to anything that
        # segments on darkness.
        y0 = int((WALL_Y - WALL_THICK / 2) * PX_PER_M)
        y1 = int((WALL_Y + WALL_THICK / 2) * PX_PER_M)
        bg[y0:y1, :, :] = 205
        for cx, w in ((MAIN_GATE_X, MAIN_GATE_W), (WEST_GATE_X, WEST_GATE_W)):
            x0 = int((cx - w / 2) * PX_PER_M)
            x1 = int((cx + w / 2) * PX_PER_M)
            bg[y0:y1, x0:x1, :] = 158
        noise = np.random.default_rng(3).normal(0, 4, bg.shape)
        return np.clip(bg.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    def render(self) -> np.ndarray:
        """A BGR frame. Agents are drawn as heads seen from directly above."""
        frame = self._bg.copy()
        if len(self.pos) == 0:
            return frame

        px = np.clip((self.pos[:, 0] * PX_PER_M).astype(int), 0, FRAME_W - 1)
        py = np.clip((self.pos[:, 1] * PX_PER_M).astype(int), 0, FRAME_H - 1)
        r = max(2, int(PERSON_R * PX_PER_M * 0.85))

        # Stamp a soft disc per agent. Vectorised over the disc offsets rather
        # than per agent, so 1400 people still render in a couple of ms.
        shades = (45 + (np.arange(len(px)) * 37 % 50)).astype(np.uint8)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                yy = np.clip(py + dy, 0, FRAME_H - 1)
                xx = np.clip(px + dx, 0, FRAME_W - 1)
                frame[yy, xx] = shades[:, None]
        return frame

    # -- ground truth -----------------------------------------------------

    def truth_points_px(self) -> np.ndarray:
        """Agent positions in image pixels, for scoring the density stage."""
        if len(self.pos) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return self.pos * PX_PER_M

    @property
    def phase_label(self) -> str:
        return phase_at(self.t).label
