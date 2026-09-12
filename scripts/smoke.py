"""Headless end-to-end run.

Drives the real pipeline against the synthetic source at wall-clock speed and
prints the per-zone numbers as they evolve, so the whole chain can be checked
without a browser.

    python -m scripts.smoke --seconds 240
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uksi import settings                       # noqa: E402
from uksi.contract import SceneConfig           # noqa: E402
from uksi.db import Store                       # noqa: E402
from uksi.pipeline import Pipeline              # noqa: E402
from uksi.store import LiveState                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--scene", default="synthetic")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-20s %(message)s")

    settings.ensure_dirs()
    scene = SceneConfig.model_validate_json(
        (settings.CONFIG_DIR / f"scene.{args.scene}.json").read_text(encoding="utf-8")
    )

    db = settings.RUNTIME_DIR / "smoke.db"
    db.unlink(missing_ok=True)
    store = Store(db)
    live = LiveState()
    pipe = Pipeline(live, store)
    pipe.apply_scene(scene)
    pipe.start()

    print(f"\nscene: {scene.name}  zones={len(scene.zones)}  "
          f"calibrated={scene.calibrated}\n")
    header = f"{'t':>5} {'zone':<22} {'p/m2':>6} {'band':<13} {'risk':<6} " \
             f"{'m/s':>5} {'trend':>7} {'ttc':>6}  headline"
    started = time.time()
    next_print = 0.0
    fired: list[str] = []

    try:
        while time.time() - started < args.seconds:
            time.sleep(0.25)
            state = live.state
            if state is None:
                continue
            for a in state.alerts:
                tag = f"{a.id}/{a.severity}/{a.escalations}"
                if tag not in fired:
                    fired.append(tag)
                    print(f"\n  *** ALERT {a.id} [{a.severity.upper()}] {a.zone_name}: "
                          f"{a.message}\n      -> {a.action}\n      evidence: {a.thumb}\n")

            elapsed = time.time() - started
            if elapsed < next_print:
                continue
            next_print = elapsed + args.every

            print(header)
            for z in state.zones:
                ttc = f"{z.time_to_critical_s:.0f}s" if z.time_to_critical_s is not None else "-"
                print(f"{elapsed:5.0f} {z.name:<22} {z.density:6.2f} {z.band:<13} "
                      f"{z.risk:<6} {z.mean_speed_ms:5.2f} {z.trend_rate_per_min:+7.2f} "
                      f"{ttc:>6}  {z.headline}")
            cor = "  ".join(f"{c.name}: {c.status} ({c.density:.1f})" for c in state.corridors)
            dbg = state.debug or {}
            print(f"      corridors: {cor}")
            print(f"      total {state.totals.count:.0f} people over "
                  f"{state.totals.area_m2:.0f} m2 | truth {dbg.get('truth_count')} "
                  f"est {dbg.get('estimated_count')} err {dbg.get('count_error_pct')}% | "
                  f"{state.proc_ms} ms/frame, {state.fps:.1f} fps, "
                  f"latency {state.latency_ms} ms | {state.model}\n")
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        print(f"\n{len(fired)} alert events in {args.seconds:.0f}s: {fired}")
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
