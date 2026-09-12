"""Single entrypoint: API and worker in one process.

    python run.py                      synthetic scene, the default
    python run.py --source clip.mp4    a video file on loop
    python run.py --source rtsp://...  a live stream

The source flag overrides the stored scene's source and nothing else, so a
calibrated scene keeps its zones when you point it at a different feed of the
same view.
"""

from __future__ import annotations

import argparse
import logging
import sys


def configure_logging(verbose: bool) -> None:
    # Windows consoles default to cp1252 and will raise on a stray non-ASCII
    # character mid-run. Force UTF-8 rather than discovering that during a demo.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("uvicorn.access", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="UKSI P-007 crowd safety analytics")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--source", default=None,
                    help="video file path, rtsp:// URL, or 'synthetic'")
    ap.add_argument("--scene", default=None,
                    help="preset id from config/scene.<id>.json to load on start")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("run")

    from uksi import settings
    from uksi.api import create_app, default_scene
    from uksi.contract import SceneConfig, SourceConfig
    from uksi.db import Store

    settings.ensure_dirs()

    # Apply CLI overrides to the stored scene before the app starts, so the
    # worker comes up on the right source rather than switching a second later.
    if args.scene or args.source:
        store = Store(settings.DB_PATH)
        if args.scene:
            path = settings.CONFIG_DIR / f"scene.{args.scene}.json"
            if not path.exists():
                log.error("no preset at %s", path)
                return 2
            scene = SceneConfig.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            scene = store.load_scene() or default_scene()

        if args.source:
            src = args.source
            if src == "synthetic":
                scene.source = SourceConfig(kind="synthetic", uri="")
            elif src.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                scene.source = SourceConfig(kind="rtsp", uri=src)
            else:
                scene.source = SourceConfig(kind="file", uri=src)
            log.info("source override: %s %s", scene.source.kind, scene.source.uri or "-")

        store.save_scene(scene)
        store.close()

    import uvicorn

    log.info("dashboard: http://%s:%d/", args.host, args.port)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
