"""Drive the demo arc and record what the control room would show.

Jumps the scenario, then watches the published state once a second and prints
every change in zone risk plus every alert. This is the close-to-real test of
the beat the pitch depends on: does a zone go amber while the scene is still
merely busy, and does it clear when the corridor opens.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    start = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    watch = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

    print(f"jumping scenario to t={start:.0f}s …")
    res = post("/api/demo/scenario", {"t_s": start})
    print(f"  -> {res}\n")

    seen_risk: dict[str, str] = {}
    seen_alerts: set[str] = set()
    t0 = time.time()

    print(f"{'elapsed':>7} {'phase':<24} {'n':>5}  zone risk changes")
    while time.time() - t0 < watch:
        try:
            s = get("/api/state")
        except Exception as exc:
            print(f"  state error: {exc}")
            time.sleep(1)
            continue

        el = time.time() - t0
        changes = []
        for z in s["zones"]:
            was = seen_risk.get(z["id"])
            if was != z["risk"]:
                seen_risk[z["id"]] = z["risk"]
                changes.append(
                    f"{z['name']}->{z['risk'].upper()} "
                    f"({z['density']:.2f} p/m2, {z['trend']}, "
                    f"{z['mean_speed_ms']:.2f} m/s)"
                )
            if z.get("time_to_critical_s") is not None and was != z["risk"]:
                changes[-1] += f" ttc={z['time_to_critical_s']:.0f}s"

        if changes:
            print(f"{el:>7.0f} {s['source']['detail'][10:]:<24} "
                  f"{s['totals']['count']:>5.0f}  " + "; ".join(changes))

        for a in s.get("alerts", []):
            if a["id"] in seen_alerts:
                continue
            seen_alerts.add(a["id"])
            print(f"\n  *** ALERT [{a['severity'].upper()}] {a['zone_name']}: {a['message']}")
            print(f"      metric : {a['metric']}")
            print(f"      action : {a['action']}")
            print(f"      thumb  : {a['thumb']}\n")

        time.sleep(1.0)

    print("\nfinal zone states:")
    s = get("/api/state")
    for z in s["zones"]:
        print(f"  {z['name']:<22} {z['density']:>5.2f} p/m2  {z['band']:<12} "
              f"{z['risk']:<6} speed {z['mean_speed_ms']:.2f}  dir {z['flow_dir_deg']}")
    print(f"\ncorridors: {[(c['name'], c['status']) for c in s['corridors']]}")
    print(f"alerts fired this run: {len(seen_alerts)}")


if __name__ == "__main__":
    main()
