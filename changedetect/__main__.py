"""Entry point: python -m changedetect [--simulate] [--port 8130]"""

from __future__ import annotations

import argparse
import logging
import threading

import uvicorn

from changedetect.config import DEFAULT_PORT, Settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Before/after imagery change detection (ISR training demo)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="web UI port (default 8130)")
    p.add_argument("--host", default="0.0.0.0", help="bind address")
    p.add_argument("--data-dir", default=None, help="reports/uploads dir (default /var/lib/change-detect)")
    p.add_argument("--simulate", action="store_true",
                   help="no-hardware demo: preloads an analysis of the bundled synthetic before/after pair")
    p.add_argument("--no-cot", action="store_true", help="disable ATAK CoT output")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    settings = Settings.from_env(args.data_dir, simulate=args.simulate)
    if args.no_cot:
        settings.cot_enabled = False

    from changedetect.web.app import create_app
    app = create_app(settings)
    log = logging.getLogger("changedetect")
    log.info("Gemini: %s | YOLO: %s | TAK %s:%d | data %s",
             f"cloud ({settings.gemini_model})" if settings.gemini_api_key else "offline",
             "on" if app.state.svc.labeler.enabled else "off", settings.tak_host, settings.tak_port, settings.data_dir)

    if args.simulate:
        def warm():
            from changedetect import pipeline
            svc = app.state.svc
            s = svc.samples[0]
            with svc.lock:
                pipeline.analyze(settings.samples_dir / s["before"], settings.samples_dir / s["after"],
                                 settings.reports_dir, pipeline.Options(), svc.labeler)
        threading.Thread(target=warm, daemon=True).start()

    log.info("Change Detect web UI at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
