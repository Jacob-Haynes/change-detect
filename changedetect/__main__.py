"""Entry point: python -m changedetect [--simulate] [--port 8130]"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

import uvicorn

from changedetect.config import DEFAULT_PORT, Settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Before/after imagery change detection (training and demo tool)")
    p.add_argument("--port", type=int, default=None, help=f"web UI port (default $WEB_PORT or {DEFAULT_PORT})")
    p.add_argument("--host", default="0.0.0.0", help="bind address")
    p.add_argument("--data-dir", default=None, help="reports/uploads dir (default /var/lib/change-detect)")
    p.add_argument("--simulate", action="store_true",
                   help="no-hardware demo: preloads an analysis of the bundled synthetic before/after pair")
    p.add_argument("--no-cot", action="store_true", help="disable ATAK CoT output")
    p.add_argument("--profile", choices=["pi", "laptop"], default=None,
                   help="pi = downscale for speed (default); laptop = full-resolution stills, GPU YOLO if present")
    p.add_argument("--tak-host", help="TAK server host (env TAK_HOST, default 127.0.0.1)")
    p.add_argument("--tak-port", type=int, help="TAK CoT TCP port (env TAK_PORT, default 8087)")
    p.add_argument("--mqtt-host", help="MQTT broker host (env MQTT_HOST, default 127.0.0.1)")
    p.add_argument("--mqtt-port", type=int, help="MQTT port (env MQTT_PORT, default 1883)")
    p.add_argument("--mqtt-user", help="MQTT user (env MQTT_USER; use isrkit from a laptop)")
    p.add_argument("--mqtt-pass", help="MQTT password (env MQTT_PASS)")
    p.add_argument("--no-heartbeat", action="store_true", help="do not publish isrkit/heartbeat/change-detect")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    settings = Settings.from_env(args.data_dir, simulate=args.simulate, profile=args.profile)
    if args.no_cot:
        settings.cot_enabled = False
    if args.no_heartbeat:
        settings.heartbeat_enabled = False
    for arg, attr in (("tak_host", "tak_host"), ("tak_port", "tak_port"), ("mqtt_host", "mqtt_host"),
                      ("mqtt_port", "mqtt_port"), ("mqtt_user", "mqtt_user"), ("mqtt_pass", "mqtt_pass")):
        if getattr(args, arg) is not None:
            setattr(settings, attr, getattr(args, arg))
    if args.port is not None:
        settings.port = args.port

    from changedetect.web.app import create_app
    app = create_app(settings)
    log = logging.getLogger("changedetect")
    svc = app.state.svc
    log.info("Profile %s | mode %s | AI: %s | YOLO: %s | TAK %s:%d | MQTT %s@%s:%d | data %s",
             settings.profile, svc.mode(), svc.advisor.source_label(), type(svc.labeler).__name__,
             settings.tak_host, settings.tak_port, settings.mqtt_user, settings.mqtt_host, settings.mqtt_port,
             settings.data_dir)

    if args.simulate:
        def warm():
            from changedetect import pipeline
            svc = app.state.svc
            s = svc.samples[0]
            with svc.lock:
                pipeline.analyze(settings.samples_dir / s["before"], settings.samples_dir / s["after"],
                                 settings.reports_dir, svc.options(), svc.labeler)
        threading.Thread(target=warm, daemon=True).start()

    log.info("Change Detect web UI at http://%s:%d", args.host, settings.port)
    # uvicorn shuts down gracefully on SIGTERM/SIGINT, then re-raises the signal into the handler
    # that was installed before it started. Make that handler a clean exit (status 0) so the
    # finally block below runs and publishes the retained "stopped" heartbeat.
    def _clean_exit(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _clean_exit)
    signal.signal(signal.SIGINT, _clean_exit)
    try:
        # uvicorn handles SIGINT/SIGTERM with a graceful shutdown and then returns here
        uvicorn.run(app, host=args.host, port=settings.port, log_level="warning", timeout_graceful_shutdown=5)
    finally:
        svc.heartbeat.stop()
        svc.cot.stop()
        log.info("Change Detect stopped")


if __name__ == "__main__":
    main()
